# -*- coding: utf-8 -*-
"""Baking LoRA/LyCORIS into an Anima model, extracting a LoRA from two models,
and converting LoRA files between 2B (28 blocks) and 2.9B (40 blocks)."""
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

from amutil import t, log, Progress, pick_device
from anima_merger import (save_with_progress, Model, DTYPES, OFFICIAL_INSERTED, OFFICIAL_SMALL_TO_LARGE,
                          parse_deep, rule_value, adapter_value, warn_rule_overlap, save, cast)

# Key endings of LoRA types -> (role)
SUFFIXES = [
    (".lora_down.weight", "down"), (".lora_up.weight", "up"),
    (".lora_A.weight", "down"), (".lora_B.weight", "up"),
    (".lora.down.weight", "down"), (".lora.up.weight", "up"),
    (".lora_mid.weight", "mid"),
    (".hada_w1_a", "h1a"), (".hada_w1_b", "h1b"), (".hada_w2_a", "h2a"), (".hada_w2_b", "h2b"),
    (".hada_t1", "ht1"), (".hada_t2", "ht2"),
    (".lokr_w1", "k1"), (".lokr_w1_a", "k1a"), (".lokr_w1_b", "k1b"),
    (".lokr_w2", "k2"), (".lokr_w2_a", "k2a"), (".lokr_w2_b", "k2b"), (".lokr_t2", "kt2"),
    (".diff", "diff"), (".diff_b", "diff_b"),
    (".dora_scale", "dora"), (".alpha", "alpha"),
]
SUFFIXES.sort(key=lambda x: -len(x[0]))
# Which role to scale to scale the whole delta of a module (for the converter)
SCALE_ROLE = ["up", "h1a", "k1", "k1a", "diff"]

STRIP_PREFIXES = ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
                  "net.", "lora_unet_", "lycoris_", "unet."]
TE_PREFIXES = ["lora_te", "text_encoder", "te_", "cond_stage_model", "qwen", "lora_te1_", "lora_te2_"]

# Different trainers' names for the same layers
ALIASES = [
    ("to_q", "q_proj"), ("to_k", "k_proj"), ("to_v", "v_proj"),
    ("to_out_0", "output_proj"), ("to_out", "output_proj"), ("o_proj", "output_proj"), ("out_proj", "output_proj"),
    ("fc1", "layer1"), ("fc2", "layer2"),
]


def flat(name):
    return re.sub(r"[._]+", "_", name).strip("_").lower()


def alias_variants(f):
    out = {f}
    for a, b in ALIASES:
        for x in list(out):
            if f"_{a}" in x or x.endswith(a):
                out.add(re.sub(rf"(^|_){a}(?=_|$)", rf"\g<1>{b}", x))
    return out


class TargetIndex:
    """Flat module name (no prefixes, '_'-joined) -> weight key in the model."""

    def __init__(self, model: Model):
        self.model = model
        self.map = {}
        for k in model.keys:
            if not k.endswith(".weight"):
                continue
            base = k[:-len(".weight")]
            nk = base
            for p in STRIP_PREFIXES:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    break
            self.map[flat(nk)] = k

    def resolve(self, lora_module):
        nk = lora_module
        for _ in range(2):
            for p in STRIP_PREFIXES:
                if nk.startswith(p):
                    nk = nk[len(p):]
        for v in alias_variants(flat(nk)):
            if v in self.map:
                return self.map[v]
        return None


def group_lora(path):
    f = safe_open(path, framework="pt", device="cpu")
    groups = {}
    unknown = 0
    for k in f.keys():
        for suf, role in SUFFIXES:
            if k.endswith(suf):
                groups.setdefault(k[:-len(suf)], {})[role] = k
                break
        else:
            unknown += 1
    return f, groups, unknown, (f.metadata() or {})


def lora_delta(f, g, shape=None, device="cpu"):
    """Full weight delta from LoRA/LoHa/LoKr parts. Returns (delta, kind). shape=None: no reshape."""
    T = lambda r: f.get_tensor(g[r]).to(device).float()
    alpha = float(f.get_tensor(g["alpha"]).float()) if "alpha" in g else None
    rs = (lambda d: d.reshape(shape)) if shape is not None else (lambda d: d)

    if "up" in g and "down" in g:
        up, down = T("up"), T("down")
        rank = down.shape[0]
        if "mid" in g:  # LoCon with tucker — almost never seen for Anima
            raise ValueError("tucker (lora_mid) is not supported")
        d = up.reshape(up.shape[0], -1) @ down.reshape(rank, -1)
        scale = (alpha / rank) if alpha is not None else 1.0
        return rs(d * scale), "lora"

    if "h1a" in g:
        if "ht1" in g:
            raise ValueError("LoHa with tucker is not supported")
        w1 = T("h1a") @ T("h1b")
        w2 = T("h2a") @ T("h2b")
        rank = f.get_slice(g["h1b"]).get_shape()[0]
        scale = (alpha / rank) if alpha is not None else 1.0
        return rs(w1 * w2 * scale), "loha"

    if "k1" in g or "k1a" in g:
        w1 = T("k1") if "k1" in g else T("k1a") @ T("k1b")
        if "kt2" in g:
            raise ValueError("LoKr with tucker is not supported")
        w2 = T("k2") if "k2" in g else T("k2a") @ T("k2b")
        rank = None
        if "k2b" in g:
            rank = f.get_slice(g["k2b"]).get_shape()[0]
        elif "k1b" in g:
            rank = f.get_slice(g["k1b"]).get_shape()[0]
        scale = (alpha / rank) if (alpha is not None and rank) else 1.0
        d = torch.kron(w1, w2.reshape(w2.shape[0], -1))
        return rs(d * scale), "lokr"

    if "diff" in g:
        return rs(T("diff")), "full"
    return None, None


def map_lora_block(idx, lora_blocks, model_blocks, m_inserted):
    """Where a LoRA block goes when LoRA and model depth differ (2B <-> 2.9B). -> [(block, coef)]"""
    if lora_blocks == model_blocks or lora_blocks == 0:
        return [(idx, 1.0)]
    if (lora_blocks, model_blocks) == (28, 40):
        res = [(OFFICIAL_SMALL_TO_LARGE[idx], 1.0)]
        if m_inserted:
            res += [(j, m_inserted) for j, src in OFFICIAL_INSERTED.items() if src == idx]
        return res
    if (lora_blocks, model_blocks) == (40, 28):
        return [(OFFICIAL_SMALL_TO_LARGE.index(idx), 1.0)] if idx in OFFICIAL_SMALL_TO_LARGE else []
    return [(idx, 1.0)]


BLOCK_IN_NAME = re.compile(r"(?:^|[._])blocks[._](\d+)[._]")


def lora_block_count(names):
    nb = 0
    for name in names:
        mm = BLOCK_IN_NAME.search(name)
        if mm and "llm_adapter" not in name:
            nb = max(nb, int(mm.group(1)) + 1)
    return nb


def cmd_bake(args):
    M = Model(args.model, "model")
    M.describe()
    idx = TargetIndex(M)
    deep, deep_default = parse_deep(args.deep)
    gen_factor = deep_default if deep_default is not None else 1.0
    warn_rule_overlap(deep, M.blocks[0])

    def rule_factor(key):
        if "llm_adapter" in key:
            return adapter_value(deep, key, gen_factor)
        m = re.search(r"blocks\.(\d+)\.(.+)$", key)
        if not m:
            return gen_factor
        return rule_value(deep, int(m.group(1)), m.group(2), gen_factor)

    deltas = {}  # model key -> accumulated delta (float32)
    for spec in args.lora:
        path, _, strength = spec.rpartition(":")
        try:
            strength = float(strength)
        except ValueError:
            path, strength = spec, 1.0
        f, groups, unknown, md = group_lora(path)
        lblocks = lora_block_count(groups)
        log(t(f"\n[LoRA] {path}  сила {strength:g}  модулей: {len(groups)}  блоков в LoRA: ~{lblocks}",
              f"\n[LoRA] {path}  strength {strength:g}  modules: {len(groups)}  blocks in LoRA: ~{lblocks}"))
        if lblocks and lblocks != M.num_blocks and (lblocks, M.num_blocks) in ((28, 40), (40, 28)):
            log(t(f"  LoRA для {lblocks} блоков, модель {M.num_blocks}: раскладываю по официальной схеме 2B<->2.9B.",
                  f"  LoRA for {lblocks} blocks, model has {M.num_blocks}: mapping by the official 2B<->2.9B layout."))
        ok = te = miss = bad = 0
        kinds = {}
        missed = []
        prog = Progress(len(groups), t("Чтение LoRA", "Reading LoRA"))
        for name, g in groups.items():
            prog.step()
            low = name.lower()
            if any(low.startswith(p) for p in TE_PREFIXES):
                te += 1
                continue
            mm = BLOCK_IN_NAME.search(name)
            targets = []
            if mm and "llm_adapter" not in name and lblocks and lblocks != M.num_blocks:
                bi = int(mm.group(1))
                for nb, coef in map_lora_block(bi, lblocks, M.num_blocks, args.m_inserted):
                    nn = name[:mm.start(1)] + str(nb) + name[mm.end(1):]
                    k = idx.resolve(nn)
                    if k:
                        targets.append((k, coef))
            else:
                k = idx.resolve(name)
                if k:
                    targets.append((k, 1.0))
            if not targets:
                miss += 1
                if len(missed) < 8:
                    missed.append(name)
                continue
            for k, coef in targets:
                shape = M.shape(k)
                try:
                    d, kind = lora_delta(f, g, shape)
                except Exception as e:
                    bad += 1
                    log(t(f"  пропуск {name}: {e}", f"  skipped {name}: {e}"))
                    break
                if d is None:
                    bad += 1
                    break
                kinds[kind] = kinds.get(kind, 0) + 1
                fac = strength * coef * rule_factor(k)
                if fac == 0:
                    continue
                d = d * fac
                deltas[k] = d if k not in deltas else deltas[k] + d
                ok += 1
            if "diff_b" in g:
                kb = idx.resolve(name)
                if kb and kb[:-7] + ".bias" in M.keys:
                    bk = kb[:-7] + ".bias"
                    db = f.get_tensor(g["diff_b"]).float() * strength
                    deltas[bk] = db if bk not in deltas else deltas[bk] + db
        log(t(f"  легло на модель: {ok}   типы: {kinds or '-'}", f"  applied to the model: {ok}   types: {kinds or '-'}"))
        if te:
            log(t(f"  слоёв текстового энкодера: {te} — пропущены (в чекпоинте Anima его нет, LoRA на TE подключай отдельно)",
                  f"  text-encoder layers: {te} — skipped (an Anima checkpoint has no TE; load TE LoRAs separately)"))
        if miss:
            log(t(f"  НЕ найдено в модели: {miss}. Примеры: ", f"  NOT found in the model: {miss}. Examples: ")
                + ", ".join(missed))
        if bad:
            log(t(f"  не удалось собрать: {bad}", f"  could not assemble: {bad}"))
        if ok == 0:
            log(t("  ВНИМАНИЕ: ни один слой не подошёл. LoRA от другой архитектуры (SDXL/Flux/…)?",
                  "  WARNING: no layer matched. A LoRA for another architecture (SDXL/Flux/…)?"))

    if not deltas:
        raise SystemExit(t("Нечего вшивать.", "Nothing to bake."))

    out = {}
    capped = 0
    rels = []
    prog = Progress(len(M.keys), t("Вшивание", "Baking"))
    for k in M.keys:
        prog.step()
        x = M.get(k)
        if k in deltas:
            w = x.float()
            d = deltas[k]
            rel = float(d.norm() / (w.norm() + 1e-12))
            if args.cap and rel > args.cap:
                d = d * (args.cap / rel)
                capped += 1
                rel = args.cap
            rels.append(rel)
            out[k] = cast(w + d, x, args.save_dtype)
        else:
            out[k] = cast(x, x, args.save_dtype)
    rels.sort()
    log(t(f"\nИзменено тензоров: {len(rels)}. Относительное изменение весов: "
          f"медиана {rels[len(rels) // 2]:.4f}, максимум {rels[-1]:.4f}",
          f"\nChanged tensors: {len(rels)}. Relative weight change: "
          f"median {rels[len(rels) // 2]:.4f}, max {rels[-1]:.4f}"))
    if args.cap:
        log(t(f"Ограничитель {args.cap:g}: подрезано тензоров {capped}", f"Limiter {args.cap:g}: tensors clipped {capped}"))
    elif rels[-1] > 0.1:
        log(t("Есть слои, изменённые больше чем на 10%. Если модель «пережарится», включи ограничитель (--cap 0.05).",
              "Some layers changed by more than 10%. If the model looks 'overcooked', enable the limiter (--cap 0.05)."))
    if args.dry_run:
        log(t("dry-run: файл не сохранён.", "dry-run: nothing saved."))
        return
    save(out, args.out, M.metadata, {"mode": "bake", "model": args.model, "loras": args.lora,
                                     "deep": args.deep, "cap": args.cap, "m_inserted": args.m_inserted})


def cmd_extract(args):
    T = Model(args.tuned, "tuned")
    B = Model(args.base, "base")
    T.describe()
    B.describe()
    if T.num_blocks != B.num_blocks:
        raise SystemExit(t("У моделей разное число блоков.", "The models have different block counts."))
    dev = pick_device(args.device)
    log(t(f"Ранг {args.rank}", f"Rank {args.rank}") +
        (t(f", ранг adaLN {args.rank_adaln}", f", adaLN rank {args.rank_adaln}") if args.rank_adaln else ""))
    deep, deep_default = parse_deep(args.deep)
    gen = deep_default if deep_default is not None else 1.0
    torch.manual_seed(0)  # svd_lowrank is randomized: fixed seed = reproducible LoRA
    out = {}
    n = skipped = 0
    kept_energy = []
    bkeys = set(B.keys)
    keys = [k for k in T.keys if k.endswith(".weight") and k in bkeys]
    prog = Progress(len(keys), t("Извлечение", "Extracting"))
    for k in keys:
        prog.step()
        if T.shape(k) != B.shape(k) or len(T.shape(k)) != 2:
            continue
        if not args.include_nonblock and ".blocks." not in "." + k and not k.startswith("blocks."):
            continue
        if "llm_adapter" in k and not args.include_nonblock:
            continue
        m = re.search(r"blocks\.(\d+)\.(.+)$", k)
        if "llm_adapter" in k:
            fac = adapter_value(deep, k, gen)
        elif m:
            fac = rule_value(deep, int(m.group(1)), m.group(2), gen)
        else:
            fac = gen
        if fac == 0:
            continue
        d = (T.get(k).float() - B.get(k).float()).to(dev) * fac
        w_norm = float(B.get(k).float().norm())
        dn = float(d.norm())
        if dn < args.min_rel * (w_norm + 1e-12):
            skipped += 1
            continue
        r = args.rank_adaln if (args.rank_adaln and "adaln" in k.lower()) else args.rank
        r = min(r, min(d.shape))
        total = float((d * d).sum())
        q = min(r + 10, min(d.shape))
        try:
            # Fast approximate SVD: only the first r components are needed (much faster than full on CPU)
            U, S, V = torch.svd_lowrank(d, q=q, niter=4)
            Vh = V.T
        except Exception:
            U, S, Vh = torch.linalg.svd(d.cpu(), full_matrices=False)
        kept_energy.append(min(1.0, float((S[:r] ** 2).sum()) / max(total, 1e-20)))
        U, S, Vh = U[:, :r], S[:r], Vh[:r]
        sq = S.sqrt()
        up = (U * sq).cpu()
        down = (sq[:, None] * Vh).cpu()
        if args.clamp:
            hi = torch.quantile(torch.cat([up.flatten(), down.flatten()]).abs()[:2_000_000], args.clamp)
            up, down = up.clamp(-hi, hi), down.clamp(-hi, hi)
        name = k[:-len(".weight")]
        for p in ("model.diffusion_model.", "diffusion_model.", "net."):
            if name.startswith(p):
                name = name[len(p):]
        name = "diffusion_model." + name
        dt = DTYPES[args.save_dtype]
        out[f"{name}.lora_up.weight"] = up.to(dt).contiguous()
        out[f"{name}.lora_down.weight"] = down.to(dt).contiguous()
        out[f"{name}.alpha"] = torch.tensor(float(r), dtype=dt)
        n += 1
    if not out:
        raise SystemExit(t("Модели почти одинаковые — нечего извлекать.", "The models are almost identical — nothing to extract."))
    kept_energy.sort()
    log(t(f"\nСлоёв в LoRA: {n}, пропущено почти без изменений: {skipped}",
          f"\nLayers in the LoRA: {n}, skipped as almost unchanged: {skipped}"))
    log(t(f"Насколько точно LoRA повторяет разницу (доля «энергии»): медиана {kept_energy[len(kept_energy) // 2]:.3f}, "
          f"худший слой {kept_energy[0]:.3f}",
          f"How well the LoRA reproduces the difference (energy share): median {kept_energy[len(kept_energy) // 2]:.3f}, "
          f"worst layer {kept_energy[0]:.3f}"))
    if kept_energy[len(kept_energy) // 2] < 0.5:
        log(t("Точность низкая: разница «размазана». Подними ранг (64–128) или это не ложится в LoRA.",
              "Low accuracy: the difference is spread out. Raise the rank (64–128), or it doesn't fit a LoRA."))
    if args.dry_run:
        log(t("dry-run: файл не сохранён.", "dry-run: nothing saved."))
        return
    md = {"anima_merger": json.dumps({"mode": "extract", "tuned": args.tuned, "base": args.base,
                                      "rank": args.rank, "deep": args.deep}, ensure_ascii=False),
          "ss_network_dim": str(args.rank), "ss_network_alpha": str(args.rank)}
    log(t(f"\nСохраняю {args.out}...", f"\nSaving {args.out}..."))
    save_with_progress(out, args.out, md)
    log(t("Готово.", "Done."))


# ------------------------------------------------------------ LoRA convert --

def convert_one(path, out_path, to_blocks, m_inserted):
    f = safe_open(path, framework="pt", device="cpu")
    keys = list(f.keys())
    md = dict(f.metadata() or {})
    modules = {}
    other = []
    for k in keys:
        for suf, role in SUFFIXES:
            if k.endswith(suf):
                modules.setdefault(k[:-len(suf)], {})[role] = (k, suf)
                break
        else:
            other.append(k)
    nb = lora_block_count(modules)
    if nb not in (28, 40):
        log(t(f"  {Path(path).name}: в LoRA {nb or 'нет'} блоков — это не Anima 2B/2.9B, пропуск.",
              f"  {Path(path).name}: the LoRA has {nb or 'no'} blocks — not Anima 2B/2.9B, skipped."))
        return False
    target = to_blocks or (40 if nb == 28 else 28)
    if target == nb:
        log(t(f"  {Path(path).name}: уже для {nb} блоков, пропуск.", f"  {Path(path).name}: already for {nb} blocks, skipped."))
        return False
    out = {}
    moved = copied = dropped = 0
    for name, parts in modules.items():
        mm = BLOCK_IN_NAME.search(name)
        if not mm or "llm_adapter" in name:
            for role, (k, suf) in parts.items():
                out[k] = f.get_tensor(k)
            continue
        bi = int(mm.group(1))
        if nb == 28:
            dests = [(OFFICIAL_SMALL_TO_LARGE[bi], 1.0)]
            if m_inserted:
                dests += [(j, m_inserted) for j, src in OFFICIAL_INSERTED.items() if src == bi]
        else:
            if bi not in OFFICIAL_SMALL_TO_LARGE:
                dropped += 1
                continue
            dests = [(OFFICIAL_SMALL_TO_LARGE.index(bi), 1.0)]
        for nbi, coef in dests:
            nn = name[:mm.start(1)] + str(nbi) + name[mm.end(1):]
            scaled = False
            for role, (k, suf) in parts.items():
                x = f.get_tensor(k)
                if coef != 1.0 and not scaled and role in SCALE_ROLE:
                    x = (x.float() * coef).to(x.dtype)
                    scaled = True
                out[nn + suf] = x
            if coef == 1.0:
                moved += 1
            else:
                copied += 1
    for k in other:
        out[k] = f.get_tensor(k)
    md["anima_merger"] = json.dumps({"mode": "lora-convert", "source": str(path), "from_blocks": nb,
                                     "to_blocks": target, "m_inserted": m_inserted}, ensure_ascii=False)
    log(t(f"  {Path(path).name}: {nb} → {target} блоков. Перенесено модулей: {moved}",
          f"  {Path(path).name}: {nb} → {target} blocks. Modules moved: {moved}")
        + (t(f", копий во вставленные блоки: {copied}", f", copies into inserted blocks: {copied}") if copied else "")
        + (t(f", выброшено (вставленные блоки 2.9B): {dropped}", f", dropped (2.9B inserted blocks): {dropped}") if dropped else ""))
    save_with_progress(out, out_path, md)
    log("  → " + out_path)
    return True


def cmd_lora_convert(args):
    files, roots = [], {}
    for p in args.lora:
        if os.path.isdir(p):
            for x in sorted(Path(p).rglob("*.safetensors"), key=lambda z: str(z).lower()):
                mm = re.match(r"^(.+)_(2\.9B|2B)$", x.stem, re.I)
                if mm and (x.parent / (mm.group(1) + ".safetensors")).exists():
                    log(t(f"  пропуск {x.name}: похоже, это уже результат конвертации",
                          f"  skipped {x.name}: looks like an earlier conversion result"))
                    continue
                files.append(str(x))
                roots[str(x)] = Path(p)
        else:
            files.append(p)
    if not files:
        raise SystemExit(t("Нет файлов LoRA.", "No LoRA files."))
    to_blocks = {"auto": None, "40": 40, "28": 28}[args.to]
    if args.out and len(files) > 1:
        raise SystemExit(t("--out только для одного файла; для нескольких используй --out-dir.",
                           "--out is for a single file; use --out-dir for several."))
    done = 0
    for i, p in enumerate(files, 1):
        log(f"\n[{i}/{len(files)}] {p}")
        if args.out:
            op = args.out
        else:
            if args.out_dir:
                # keep the subfolder layout of a scanned folder inside --out-dir
                rel = Path(p).parent.relative_to(roots[p]) if p in roots else Path()
                folder = Path(args.out_dir) / rel
            else:
                folder = Path(p).parent
            folder.mkdir(parents=True, exist_ok=True)
            suffix = args.suffix
            if not suffix:
                nb = lora_block_count(safe_open(p, framework="pt", device="cpu").keys())
                suffix = "_2.9B" if (to_blocks or (40 if nb == 28 else 28)) == 40 else "_2B"
            op = str(folder / (Path(p).stem + suffix + ".safetensors"))
        if os.path.abspath(op) == os.path.abspath(p):
            raise SystemExit(t("Выходной файл совпадает с исходным — выбери другое имя.",
                               "The output file equals the input — choose another name."))
        if convert_one(p, op, to_blocks, args.m_inserted):
            done += 1
    log(t(f"\nГотово: сконвертировано {done} из {len(files)}.", f"\nDone: converted {done} of {len(files)}."))


# --------------------------------------------------------------- LoRA merge --

def _canon(name):
    """Format-independent module name: 'blocks_3_self_attn_q_proj'."""
    nk = name
    for _ in range(2):
        for p in STRIP_PREFIXES:
            if nk.startswith(p):
                nk = nk[len(p):]
    f = flat(nk)
    for a, b in ALIASES:
        f = re.sub(rf"(^|_){a}(?=_|$)", rf"\g<1>{b}", f)
    return f


def _merge_factor(rules, default, name):
    """Per-block / adapter rule value for a LoRA module name (dots and underscores both match)."""
    from anima_merger import ADAPTER
    n2 = name.replace(".", "_")
    if "llm_adapter" in name:
        rest = n2.split("llm_adapter_", 1)[-1]
        val = None
        for lo, hi, pats, v in rules:
            if lo == ADAPTER and any(p in ("*", "all") or p.replace(".", "_").lstrip("^") in rest for p in pats):
                val = v
        return default if val is None else val
    mm = BLOCK_IN_NAME.search(name)
    if not mm:
        return default
    bi = int(mm.group(1))
    rest = n2[mm.end(1):].lstrip("_")
    val = None
    for lo, hi, pats, v in rules:
        if lo == ADAPTER or not (lo <= bi <= hi):
            continue
        for p in pats:
            q = p.replace(".", "_")
            if p in ("*", "all") or (rest.startswith(q[1:]) if q.startswith("^") else q in rest):
                val = v
    return default if val is None else val


def cmd_lora_merge(args):
    """Mix several LoRAs into one LoRA file: concat (exact) or SVD (to a chosen rank)."""
    dev = pick_device(args.device)
    torch.manual_seed(0)
    deep, deep_default = parse_deep(args.deep)
    gen = deep_default if deep_default is not None else 1.0
    srcs = []
    for spec in args.lora:
        path, _, st = spec.rpartition(":")
        try:
            st = float(st)
        except ValueError:
            path, st = spec, 1.0
        f, groups, unknown, md = group_lora(path)
        groups = {n: g for n, g in groups.items() if not any(n.lower().startswith(p) for p in TE_PREFIXES)}
        nb = lora_block_count(groups)
        kinds = {("lora" if ("up" in g and "down" in g and "mid" not in g) else "other") for g in groups.values()}
        log(t(f"[LoRA] {path}  сила {st:g}  модулей: {len(groups)}  блоков: {nb}",
              f"[LoRA] {path}  strength {st:g}  modules: {len(groups)}  blocks: {nb}"))
        srcs.append({"path": path, "f": f, "groups": groups, "st": st, "nb": nb, "kinds": kinds, "meta": md,
                     "canon": {_canon(n): n for n in groups}})
    nbs = {s["nb"] for s in srcs if s["nb"]}
    if len(nbs) > 1:
        raise SystemExit(t(f"LoRA для разных архитектур (блоков: {sorted(nbs)}). Сначала приведи их к одной "
                           f"вкладкой «Конвертер LoRA 2B ↔ 2.9B».",
                           f"LoRAs for different architectures (blocks: {sorted(nbs)}). Convert them to one "
                           f"architecture first with the 'LoRA converter 2B ↔ 2.9B' tab."))
    mode = args.mode
    all_plain = all(s["kinds"] == {"lora"} for s in srcs)
    if mode == "auto":
        mode = "concat" if all_plain else "svd"
    if mode == "concat" and not all_plain:
        raise SystemExit(t("concat работает только для обычных LoRA (не LoHa/LoKr). Выбери svd.",
                           "concat works only for plain LoRAs (not LoHa/LoKr). Use svd."))
    log(t(f"Режим: {mode}", f"Mode: {mode}") + (f", rank {args.rank}" if mode == "svd" else ""))
    # output naming follows the first LoRA that has the module
    canon_all = []
    seen = set()
    for s in srcs:
        for c in s["canon"]:
            if c not in seen:
                seen.add(c)
                canon_all.append(c)
    dt = DTYPES[args.save_dtype]
    # Different key formats (kohya "lora_unet_..." vs "diffusion_model.blocks...") -> write everything as kohya,
    # which can always be built from the canonical name.
    fmts = {n.startswith("lora_unet_") for s in srcs for n in s["groups"]}
    unify = len(fmts) > 1
    if unify:
        log(t("LoRA в разных форматах — сохраняю в едином формате kohya (lora_unet_...).",
              "LoRAs in different formats — saving in one kohya format (lora_unet_...)."))
    out = {}
    kept, total_rank = [], 0
    prog = Progress(len(canon_all), t("Смешивание LoRA", "Merging LoRAs"))
    for c in canon_all:
        prog.step()
        owner = next(s for s in srcs if c in s["canon"])
        oname = ("lora_unet_" + c) if unify else owner["canon"][c]
        oname_src = owner["canon"][c]
        parts = []
        for s in srcs:
            n = s["canon"].get(c)
            if n is None:
                continue
            fac = s["st"] * _merge_factor(deep, gen, n)
            if fac == 0:
                continue
            parts.append((s, n, fac))
        if not parts:
            continue
        if mode == "concat":
            ups, downs = [], []
            for s, n, fac in parts:
                g = s["groups"][n]
                up = s["f"].get_tensor(g["up"]).float()
                down = s["f"].get_tensor(g["down"]).float()
                r = down.shape[0]
                alpha = float(s["f"].get_tensor(g["alpha"]).float()) if "alpha" in g else float(r)
                ups.append(up.reshape(up.shape[0], -1) * (fac * alpha / r))
                downs.append(down.reshape(r, -1))
            shapes = {(u.shape[0], d.shape[1]) for u, d in zip(ups, downs)}
            if len(shapes) > 1:
                log(t(f"  пропуск {oname}: разные формы {shapes}", f"  skipped {oname}: different shapes {shapes}"))
                continue
            up = torch.cat(ups, 1)
            down = torch.cat(downs, 0)
            r = down.shape[0]
            g0 = owner["groups"][oname_src]
            up_like = owner["f"].get_slice(g0["up"]).get_shape()
            down_like = owner["f"].get_slice(g0["down"]).get_shape()
            if len(up_like) == 4:          # conv-style LoRA keeps its 4D layout
                up = up.reshape(up.shape[0], r, 1, 1)
                down = down.reshape(r, *down_like[1:])
            us, ds = (".lora_up.weight", ".lora_down.weight") if unify else (_suffix(g0["up"]), _suffix(g0["down"]))
            out[oname + us] = up.to(dt).contiguous()
            out[oname + ds] = down.to(dt).contiguous()
            out[oname + ".alpha"] = torch.tensor(float(r), dtype=dt)
            total_rank = max(total_rank, r)
        else:
            acc = None
            for s, n, fac in parts:
                d, kind = lora_delta(s["f"], s["groups"][n], None, dev)
                if d is None:
                    continue
                d = d.reshape(d.shape[0], -1) * fac
                if acc is not None and acc.shape != d.shape:
                    log(t(f"  пропуск {oname}: разные формы", f"  skipped {oname}: different shapes"))
                    acc = None
                    break
                acc = d if acc is None else acc + d
            if acc is None:
                continue
            r = min(args.rank, min(acc.shape))
            tot = float((acc * acc).sum())
            if tot <= 0:
                continue
            try:
                U, S, V = torch.svd_lowrank(acc, q=min(r + 10, min(acc.shape)), niter=4)
                Vh = V.T
            except Exception:
                U, S, Vh = torch.linalg.svd(acc.cpu(), full_matrices=False)
            kept.append(min(1.0, float((S[:r] ** 2).sum()) / tot))
            sq = S[:r].sqrt()
            up = (U[:, :r] * sq).cpu()
            down = (sq[:, None] * Vh[:r]).cpu()
            g0 = owner["groups"][oname_src]
            us = _suffix(g0["up"]) if ("up" in g0 and not unify) else ".lora_up.weight"
            ds = _suffix(g0["down"]) if ("down" in g0 and not unify) else ".lora_down.weight"
            out[oname + us] = up.to(dt).contiguous()
            out[oname + ds] = down.to(dt).contiguous()
            out[oname + ".alpha"] = torch.tensor(float(r), dtype=dt)
            total_rank = max(total_rank, r)
    if not out:
        raise SystemExit(t("Нечего смешивать.", "Nothing to merge."))
    log(t(f"\nМодулей в итоговой LoRA: {len([k for k in out if k.endswith('.alpha')])}, ранг до {total_rank}",
          f"\nModules in the merged LoRA: {len([k for k in out if k.endswith('.alpha')])}, rank up to {total_rank}"))
    if kept:
        kept.sort()
        log(t(f"Точность SVD (доля «энергии»): медиана {kept[len(kept) // 2]:.3f}, худший слой {kept[0]:.3f}",
              f"SVD accuracy (energy share): median {kept[len(kept) // 2]:.3f}, worst layer {kept[0]:.3f}"))
        if kept[len(kept) // 2] < 0.9:
            log(t("Смесь не помещается в этот ранг — подними --rank (например, до суммы рангов) или выбери concat.",
                  "The mix doesn't fit this rank — raise --rank (e.g. to the sum of ranks) or use concat."))
    else:
        log(t("concat: смесь точная, без потерь (ранг = сумма рангов).",
              "concat: the mix is exact, lossless (rank = sum of ranks)."))
    if args.dry_run:
        log(t("dry-run: файл не сохранён.", "dry-run: nothing saved."))
        return
    md = {"anima_merger": json.dumps({"mode": "lora-merge", "loras": args.lora, "deep": args.deep,
                                      "merge_mode": mode, "rank": args.rank if mode == "svd" else total_rank},
                                     ensure_ascii=False),
          "ss_network_dim": str(total_rank), "ss_network_alpha": str(total_rank)}
    for k in ("ss_network_module", "ss_base_model_version", "modelspec.architecture"):
        if k in srcs[0]["meta"]:
            md[k] = srcs[0]["meta"][k]
    log(t(f"\nСохраняю {args.out}...", f"\nSaving {args.out}..."))
    save_with_progress(out, args.out, md)
    log(t("Готово.", "Done."))


def _suffix(key):
    for suf, role in SUFFIXES:
        if key.endswith(suf):
            return suf
    return ""


def add_parsers(sub):
    p = sub.add_parser("bake", help=t("вшить одну или несколько LoRA/LyCORIS в модель",
                                      "bake one or more LoRA/LyCORIS into a model"))
    p.add_argument("--model", required=True)
    p.add_argument("--lora", action="append", required=True,
                   help=t("путь[:сила], можно несколько раз: --lora style.safetensors:0.7 --lora char.safetensors:0.8",
                          "path[:strength], repeatable: --lora style.safetensors:0.7 --lora char.safetensors:0.8"))
    p.add_argument("--out")
    p.add_argument("--deep", default="",
                   help=t("множители по блокам/слоям, как в merge: '1, L00-L05:*:0.5, L20-L27:mlp:0, adapter:*:0'",
                          "per-block/layer multipliers, as in merge: '1, L00-L05:*:0.5, L20-L27:mlp:0, adapter:*:0'"))
    p.add_argument("--cap", type=float, default=0.0,
                   help=t("ограничитель: макс. относительное изменение одного тензора (0 = выкл, 0.03–0.08 = мягко)",
                          "limiter: max relative change of one tensor (0 = off, 0.03–0.08 = gentle)"))
    p.add_argument("--m-inserted", type=float, default=0.0,
                   help=t("LoRA от 2B на модели 2.9B: доля для вставленных блоков (0 = не трогать)",
                          "2B LoRA on a 2.9B model: share for inserted blocks (0 = leave alone)"))
    p.add_argument("--save-dtype", choices=list(DTYPES), default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_bake)

    p = sub.add_parser("extract", help=t("извлечь LoRA как разницу двух моделей (SVD)",
                                         "extract a LoRA as the difference of two models (SVD)"))
    p.add_argument("--tuned", required=True, help=t("модель со стилем/концептами", "model with the style/concepts"))
    p.add_argument("--base", required=True, help=t("модель, от которой считать разницу", "model to subtract"))
    p.add_argument("--out")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--rank-adaln", type=int, default=0, help=t("отдельный ранг для adaLN (0 = как --rank)",
                                                               "separate rank for adaLN (0 = same as --rank)"))
    p.add_argument("--deep", default="", help=t("какие блоки/слои брать и с каким множителем",
                                                "which blocks/layers to take, with multipliers"))
    p.add_argument("--include-nonblock", action="store_true", help=t("включить адаптер и не-блочные слои",
                                                                     "include the adapter and non-block layers"))
    p.add_argument("--min-rel", type=float, default=1e-4, help=t("пропускать слои, изменённые меньше этого",
                                                                 "skip layers changed less than this"))
    p.add_argument("--clamp", type=float, default=0.0, help=t("обрезать выбросы по квантилю (0.99) — 0 = выкл",
                                                              "clip outliers at this quantile (0.99) — 0 = off"))
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("lora-convert", help=t("сконвертировать LoRA 2B <-> 2.9B (файл в файл)",
                                              "convert LoRA files 2B <-> 2.9B"))
    p.add_argument("--lora", nargs="+", required=True, help=t("файлы LoRA или папка", "LoRA files or a folder"))
    p.add_argument("--to", choices=["auto", "40", "28"], default="auto",
                   help=t("в какую архитектуру: auto = в противоположную", "target architecture: auto = the other one"))
    p.add_argument("--m-inserted", type=float, default=0.0,
                   help=t("2B → 2.9B: копия модуля во вставленные блоки с этим множителем (0 = оставить их пустыми)",
                          "2B → 2.9B: copy modules into inserted blocks with this factor (0 = leave them empty)"))
    p.add_argument("--out", default=None, help=t("выходной файл (для одного входного)", "output file (single input)"))
    p.add_argument("--out-dir", default=None, help=t("папка для результатов (по умолчанию рядом с исходником)",
                                                     "output folder (default: next to the source)"))
    p.add_argument("--suffix", default="", help=t("суффикс имени (по умолчанию _2.9B / _2B)",
                                                  "name suffix (default _2.9B / _2B)"))
    p.set_defaults(func=cmd_lora_convert)

    p = sub.add_parser("lora-merge", help=t("смешать несколько LoRA в одну", "mix several LoRAs into one"))
    p.add_argument("--lora", action="append", required=True,
                   help=t("путь[:сила], можно несколько раз", "path[:strength], repeatable"))
    p.add_argument("--out")
    p.add_argument("--mode", choices=["auto", "concat", "svd"], default="auto",
                   help=t("concat = точно, ранг = сумма рангов; svd = сжать до --rank; auto = concat, если возможно",
                          "concat = exact, rank = sum of ranks; svd = compress to --rank; auto = concat when possible"))
    p.add_argument("--rank", type=int, default=64, help=t("svd: ранг результата", "svd: rank of the result"))
    p.add_argument("--deep", default="", help=t("множители по блокам/слоям: '1, L00-L13:*:0, L14-L27:*:1'",
                                                "per-block/layer multipliers: '1, L00-L13:*:0, L14-L27:*:1'"))
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_lora_merge)
