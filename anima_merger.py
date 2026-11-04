#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anima Merger — merging and porting fine-tunes between Anima 2B (28 blocks) and Anima 2.9B (40 blocks).

Commands:
  merge   — A + (B - C) * M for models of the same architecture (incl. rebase).
  map     — build a 2B <-> 2.9B block map (auto / spec / compare) and save it as JSON.
  cross   — port a fine-tune between architectures using a block map.
  info    — show model structure (prefix, block count, layer types).
  bake    — bake LoRA/LoHa/LoKr into a model (several at once, per block, with a limiter).
  extract — extract a LoRA from the difference of two models (SVD).
  lora-convert — convert LoRA files 2B <-> 2.9B.
  parent / family / genome / blocks / meta / lora-info — analyzers with HTML reports.

Usage: python anima_merger.py <command> --help
"""
import argparse
import json
import os
import re
import sys
import time
import zlib
from collections import Counter, defaultdict

from amutil import t, log, Progress, safe_print_setup, ltype

try:
    import torch
    from safetensors import safe_open
except ImportError:
    print("pip install torch safetensors")
    sys.exit(1)

BLOCK_RE = re.compile(r"^(?P<prefix>.*?)blocks\.(?P<idx>\d+)\.(?P<rest>.+)$")

# Output projections: they were zeroed in inserted layers during expansion,
# so they tell an "original" block from a "copy" best.
DEFAULT_OUT_PATTERNS = ["output_proj", "out_proj", "to_out", "proj_out", "o_proj",
                        "mlp.layer2", "fc2", "linear2"]
DEFAULT_ADALN_PATTERNS = ["adaln", "adaLN", "modulation"]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Official Anima 2B (28) -> 2.9B (40) expansion: positions of inserted blocks in the
# 40-block model and the 2B block each of them was copied from.
OFFICIAL_INSERTED = {2: 1, 5: 3, 8: 5, 11: 7, 14: 9, 17: 11, 21: 14, 24: 16, 27: 18, 30: 20, 33: 22, 36: 24}
OFFICIAL_SMALL_TO_LARGE = [j for j in range(40) if j not in OFFICIAL_INSERTED]


def matches_any(name, patterns):
    low = name.lower()
    return any(p.lower() in low for p in patterns)


# ---------------------------------------------------------------- model ----

class Model:
    """Lazy .safetensors wrapper: tensors are read one at a time."""

    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.f = safe_open(path, framework="pt", device="cpu")
        self.keys = list(self.f.keys())
        self.metadata = self.f.metadata() or {}

        per_prefix = defaultdict(set)
        for k in self.keys:
            m = BLOCK_RE.match(k)
            if m:
                per_prefix[m["prefix"]].add(int(m["idx"]))
        if not per_prefix:
            raise SystemExit(t(f"[{label}] не найдено ключей вида '...blocks.N....' в {path}",
                               f"[{label}] no '...blocks.N....' keys found in {path}"))
        # Main DiT = prefix with the most blocks (the adapter has its own blocks).
        self.prefix = max(per_prefix, key=lambda p: len(per_prefix[p]))
        idxs = sorted(per_prefix[self.prefix])
        self.num_blocks = idxs[-1] + 1
        if len(idxs) != self.num_blocks:
            log(t(f"[{label}] ВНИМАНИЕ: номера блоков идут с пропусками: {idxs}",
                  f"[{label}] WARNING: block numbers have gaps: {idxs}"))

        self.blocks = defaultdict(dict)   # idx -> rest -> full key
        self.other = {}                   # normalized key -> full key
        for k in self.keys:
            m = BLOCK_RE.match(k)
            if m and m["prefix"] == self.prefix:
                self.blocks[int(m["idx"])][m["rest"]] = k
            else:
                nk = k[len(self.prefix):] if self.prefix and k.startswith(self.prefix) else k
                self.other[nk] = k

    def get(self, key):
        return self.f.get_tensor(key)

    def shape(self, key):
        return tuple(self.f.get_slice(key).get_shape())

    def block(self, idx, rest):
        k = self.blocks.get(idx, {}).get(rest)
        return None if k is None else self.get(k)

    def describe(self):
        log(f"[{self.label}] {self.path}")
        log(t(f"  префикс DiT: '{self.prefix}'   блоков: {self.num_blocks}   всего ключей: {len(self.keys)}",
              f"  DiT prefix: '{self.prefix}'   blocks: {self.num_blocks}   keys: {len(self.keys)}"))


# ---------------------------------------------------------- multipliers --

def parse_block_m(spec):
    """'0-9:0.3,10-39:0.6,20:1' -> {idx: factor}"""
    res = {}
    if not spec:
        return res
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, val = part.split(":")
        val = float(val)
        if "-" in rng:
            a, b = rng.split("-")
            for i in range(int(a), int(b) + 1):
                res[i] = val
        else:
            res[int(rng)] = val
    return res


ADAPTER = -1  # rule "block" index for llm_adapter rules


def parse_deep(spec):
    """Rules as in chattiori / Forge Neo recipes:
    'L16-L27:mlp.layer1 mlp.layer2:0.6, L00-L27:adaln:0'  — blocks, layer-name substrings, value.
    A bare number first ('0, L16-L27:mlp:0.6') sets the default M. The last matching rule wins.
    'adapter:*:0' / 'adapter:self_attn:0.5' — rules for the LLM adapter (llm_adapter.*).
    A pattern starting with '^' matches only at the start of the layer name ('^cross_attn')."""
    rules, default = [], None
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"-?[\d.]+", part):
            default = float(part)
            continue
        m = re.fullmatch(r"(?:llm_)?adapter\s*:\s*(.+?)\s*:\s*(-?[\d.]+)", part, re.I)
        if m:
            rules.append((ADAPTER, ADAPTER, m.group(1).split(), float(m.group(2))))
            continue
        m = re.fullmatch(r"L?(\d+)(?:\s*-\s*L?(\d+))?\s*:\s*(.+?)\s*:\s*(-?[\d.]+)", part)
        if not m:
            raise SystemExit(t(f"Не понял правило --deep: '{part}' (нужно вида L16-L27:mlp.layer1 mlp.layer2:0.6 "
                               f"или adapter:*:0)",
                               f"Can't parse --deep rule: '{part}' (expected L16-L27:mlp.layer1 mlp.layer2:0.6 "
                               f"or adapter:*:0)"))
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        rules.append((lo, hi, m.group(3).split(), float(m.group(4))))
    return rules, default


def pat_match(p, rest):
    if p in ("*", "all"):
        return True
    if p.startswith("^"):
        return rest.startswith(p[1:])
    return p in rest


def rule_value(rules, idx, rest, default):
    """Value for a block layer (idx, rest) or `default` if no rule matches."""
    val = None
    for lo, hi, pats, v in rules:
        if lo != ADAPTER and lo <= idx <= hi and any(pat_match(p, rest) for p in pats):
            val = v
    return default if val is None else val


def adapter_value(rules, key, default):
    """Value for an llm_adapter key or `default`."""
    if "llm_adapter" not in key:
        return default
    rest = key.split("llm_adapter.", 1)[-1]
    val = None
    for lo, hi, pats, v in rules:
        if lo == ADAPTER and any(pat_match(p, rest) for p in pats):
            val = v
    return default if val is None else val


def warn_rule_overlap(rules, rests):
    """Warn when a rule pattern catches adaLN layers by accident
    (e.g. 'cross_attn' also matches 'adaln_modulation_cross_attn')."""
    rests = list(rests)
    seen = set()
    for lo, hi, pats, v in rules:
        if lo == ADAPTER:
            continue
        for p in pats:
            if p in ("*", "all") or p.startswith("^") or "adaln" in p.lower() or "modulation" in p.lower() or p in seen:
                continue
            hit_adaln = [r for r in rests if pat_match(p, r) and ltype(r) == "adaln"]
            hit_other = [r for r in rests if pat_match(p, r) and ltype(r) != "adaln"]
            if hit_adaln and hit_other:
                seen.add(p)
                log(t(f"ВНИМАНИЕ: правило «{p}» цепляет и слои модуляции adaLN, например «{hit_adaln[0]}». "
                      f"Если они не нужны, пиши «^{p}» (только с начала имени).",
                      f"WARNING: rule '{p}' also matches adaLN modulation layers, e.g. '{hit_adaln[0]}'. "
                      f"If you don't want them, write '^{p}' (match from the start of the name only)."))


class Mults:
    def __init__(self, args):
        self.deep, deep_default = parse_deep(getattr(args, "deep", ""))
        self.m = deep_default if deep_default is not None else args.m
        self.m_adaln = self.m if args.m_adaln is None else args.m_adaln
        self.m_nonblock = self.m if args.m_nonblock is None else args.m_nonblock
        self.block_factor = parse_block_m(args.block_m)
        self.adaln_patterns = args.adaln_patterns

    def for_block(self, target_idx, rest):
        base = self.m_adaln if matches_any(rest, self.adaln_patterns) else self.m
        return rule_value(self.deep, target_idx, rest, base * self.block_factor.get(target_idx, 1.0))

    def for_other(self, key):
        base = self.m_adaln if matches_any(key, self.adaln_patterns) else self.m_nonblock
        return adapter_value(self.deep, key, base)


class DeltaStats:
    """Sanity check of the subtracted base: |B-C| / |B|."""

    def __init__(self):
        self.d2 = 0.0
        self.b2 = 0.0
        self.n = 0

    def add(self, b, c):
        d = (b - c)
        self.d2 += float((d * d).sum())
        self.b2 += float((b * b).sum())
        self.n += 1

    def report(self, what):
        if not self.n or self.b2 == 0:
            return
        r = (self.d2 / self.b2) ** 0.5
        log(t(f"\nОтносительный размер дельты {what}: {r:.4f}", f"\nRelative delta size {what}: {r:.4f}"))
        if r < 1e-6:
            log(t("  Дельта нулевая: файнтюн совпадает с базой (или вычли ту же модель).",
                  "  The delta is zero: the fine-tune equals the base (or the same model was subtracted)."))
        elif r > 0.5:
            log(t("  ВНИМАНИЕ: дельта очень большая. Похоже, база выбрана неверно "
                  "(файнтюн обучался не от неё). Результат, скорее всего, будет сломан.",
                  "  WARNING: the delta is very large. The base is probably wrong "
                  "(the fine-tune wasn't trained from it). The result will likely be broken."))
        elif r > 0.2:
            log(t("  Дельта крупная: возможно, база не та или файнтюн очень глубокий.",
                  "  The delta is large: maybe the wrong base, or a very deep fine-tune."))
        else:
            log(t("  Похоже на нормальный файнтюн относительно этой базы.",
                  "  Looks like a normal fine-tune of this base."))


_ST_DTYPE = {
    "float32": "F32", "float16": "F16", "bfloat16": "BF16", "float64": "F64",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8", "uint8": "U8", "bool": "BOOL",
    "float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2",
}


def save_with_progress(tensors, out_path, metadata=None):
    """Write .safetensors one tensor at a time with progress (safetensors' save_file builds
    the whole file in memory and stays silent). Writes <name>.part, renames at the end."""
    names = list(tensors)
    header, off = {}, 0
    for n in names:
        tt = tensors[n]
        dt = _ST_DTYPE.get(str(tt.dtype).replace("torch.", ""))
        if dt is None:
            raise ValueError(f"Unsupported dtype {tt.dtype} in {n}")
        nb = tt.numel() * tt.element_size()
        header[n] = {"dtype": dt, "shape": list(tt.shape), "data_offsets": [off, off + nb]}
        off += nb
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    total = off
    tmp = out_path + ".part"
    t0 = time.time()
    prog = Progress(max(1, total // (1 << 20)), t("Запись файла", "Writing file"))
    done_mb = 0
    done = 0
    try:
        with open(tmp, "wb") as f:
            f.write(len(hb).to_bytes(8, "little"))
            f.write(hb)
            for n in names:
                tt = tensors[n].detach().contiguous().reshape(-1)
                if tt.numel():
                    f.write(memoryview(tt.view(torch.uint8).numpy()))
                done += tt.numel() * tt.element_size()
                mb = done >> 20
                if mb > done_mb:
                    prog.step(mb - done_mb)
                    done_mb = mb
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    prog.finish()
    el = max(time.time() - t0, 1e-6)
    log(t(f"  записано {total / 2**30:.2f} ГБ за {el:.0f} с ({total / 2**20 / el:.0f} МБ/с)",
          f"  wrote {total / 2**30:.2f} GB in {el:.0f} s ({total / 2**20 / el:.0f} MB/s)"))


def save(out_tensors, out_path, metadata, info):
    md = dict(metadata or {})
    md["anima_merger"] = json.dumps(info, ensure_ascii=False)
    md = {str(k): str(v) for k, v in md.items()}
    log(t(f"\nСохраняю {out_path} ({len(out_tensors)} тензоров)...",
          f"\nSaving {out_path} ({len(out_tensors)} tensors)..."))
    save_with_progress(out_tensors, out_path, md)
    log(t("Готово.", "Done."))


def cast(x, like, save_dtype):
    return x.to(DTYPES[save_dtype] if save_dtype else like.dtype).contiguous()


# ----------------------------------------------------------- delta modes --

MODES = ["add", "topk", "dare", "ortho", "ortho_delta", "normdir"]


def apply_mode(a, d, m, mode, density=0.2, gen=None, c=None, proj=1.0):
    """a — weights of A (float), d — delta B-C (float), m — multiplier. Returns new weights.
    c — weights of C (float), needed only for ortho_delta; proj — share of the parallel part to remove."""
    if mode == "add" or d.numel() < 2:
        return a + m * d
    if mode == "topk":
        # Only the strongest changes: the `density` share of elements with the largest |d|.
        k = max(1, int(d.numel() * density))
        if k >= d.numel():
            return a + m * d
        flat = d.abs().flatten()
        thr = torch.kthvalue(flat, flat.numel() - k + 1).values
        return a + m * d * (d.abs() >= thr)
    if mode == "dare":
        # Randomly keep a `density` share of the delta and rescale by 1/density.
        keep = torch.rand(d.shape, generator=gen) < density
        return a + m * d * keep / max(density, 1e-6)
    if mode == "ortho":
        # Remove the part of the delta parallel to A: add only what's "new".
        if a.dim() >= 2:
            a2, d2 = a.reshape(a.shape[0], -1), d.reshape(d.shape[0], -1)
            proj = (d2 * a2).sum(1, keepdim=True) / (a2.pow(2).sum(1, keepdim=True) + 1e-12)
            return a + m * (d2 - proj * a2).reshape(a.shape)
        proj = (d * a).sum() / (a.pow(2).sum() + 1e-12)
        return a + m * (d - proj * a)
    if mode == "ortho_delta":
        # Remove the part of the delta parallel to A's own change (A - C), row by row:
        # whatever B pushes in the same (or opposite) direction as A already went is dropped,
        # only the genuinely new part is added. Where A == C (e.g. untouched blocks) nothing is removed.
        r = a - c
        if a.dim() >= 2:
            r2, d2 = r.reshape(r.shape[0], -1), d.reshape(d.shape[0], -1)
            coef = (d2 * r2).sum(1, keepdim=True) / (r2.pow(2).sum(1, keepdim=True) + 1e-12)
            return a + m * (d2 - proj * coef * r2).reshape(a.shape)
        coef = (d * r).sum() / (r.pow(2).sum() + 1e-12)
        return a + m * (d - proj * coef * r)
    if mode == "normdir":
        # Direction as in a plain merge, row length = blend of the lengths of A and A+d.
        w = a + m * d
        if a.dim() < 2:
            return w
        a2, w2, f2 = a.reshape(a.shape[0], -1), w.reshape(w.shape[0], -1), (a + d).reshape(a.shape[0], -1)
        mag = (1 - m) * a2.norm(dim=1, keepdim=True) + m * f2.norm(dim=1, keepdim=True)
        return (w2 / (w2.norm(dim=1, keepdim=True) + 1e-12) * mag).reshape(a.shape)
    raise SystemExit(f"Unknown mode {mode}")


# ------------------------------------------------------------------ merge --

def cmd_merge(args):
    A = Model(args.a, "A")
    B = Model(args.b, "B")
    C = Model(args.c, "C")
    for x in (A, B, C):
        x.describe()
    if not (A.num_blocks == B.num_blocks == C.num_blocks):
        raise SystemExit(t("Число блоков у A, B и C разное. Для 2B <-> 2.9B используй команду cross.",
                           "A, B and C have different block counts. Use the cross command for 2B <-> 2.9B."))

    mult = Mults(args)
    warn_rule_overlap(mult.deep, A.blocks[0])
    stats = DeltaStats()
    out = {}
    changed = skipped = 0
    gen = torch.Generator().manual_seed(args.seed)
    if args.mode != "add":
        log(t("Режим: ", "Mode: ") + args.mode + (f", density={args.density}" if args.mode in ("topk", "dare") else "")
            + (f", proj={args.proj}" if args.mode == "ortho_delta" else ""))

    # Match keys relative to the prefix, so different prefixes don't get in the way.
    def norm(model):
        d = {}
        for i, rests in model.blocks.items():
            for r, k in rests.items():
                d[("b", i, r)] = k
        for nk, k in model.other.items():
            d[("o", nk)] = k
        return d

    nA, nB, nC = norm(A), norm(B), norm(C)
    prog = Progress(len(nA), t("Мердж", "Merge"))
    for nk, ka in nA.items():
        prog.step()
        ta = A.get(ka)
        kb, kc = nB.get(nk), nC.get(nk)
        m = mult.for_block(nk[1], nk[2]) if nk[0] == "b" else mult.for_other(nk[1])
        if kb is None or kc is None or m == 0:
            out[ka] = cast(ta, ta, args.save_dtype)
            if kb is None or kc is None:
                skipped += 1
            continue
        tb, tc = B.get(kb), C.get(kc)
        if not (ta.shape == tb.shape == tc.shape):
            log(t("  форма не совпадает, пропуск: ", "  shape mismatch, skipped: ") +
                f"{ka} {tuple(ta.shape)} {tuple(tb.shape)} {tuple(tc.shape)}")
            out[ka] = cast(ta, ta, args.save_dtype)
            skipped += 1
            continue
        tbf, tcf = tb.float(), tc.float()
        if nk[0] == "b":
            stats.add(tbf, tcf)
        res = apply_mode(ta.float(), tbf - tcf, m, args.mode, args.density, gen, c=tcf, proj=args.proj) \
            if ta.dim() >= 2 else ta.float() + m * (tbf - tcf)
        out[ka] = cast(res, ta, args.save_dtype)
        changed += 1

    stats.report("B - C")
    log(t(f"Изменено тензоров: {changed}, оставлено как в A (нет пары/форма/M=0): {skipped}",
          f"Changed tensors: {changed}, kept from A (no pair/shape/M=0): {skipped}"))
    if args.dry_run:
        log(t("dry-run: файл не сохранён.", "dry-run: nothing saved."))
        return
    save(out, args.out, A.metadata, {
        "mode": "merge", "formula": "A + (B - C) * M", "delta_mode": args.mode, "density": args.density,
        "proj": args.proj if args.mode == "ortho_delta" else None,
        "A": args.a, "B": args.b, "C": args.c, "M": args.m,
        "m_adaln": mult.m_adaln, "m_nonblock": mult.m_nonblock, "block_m": args.block_m, "deep": args.deep})


# -------------------------------------------------------------- block map -----

def _sample(x, n, seed):
    v = x.float().flatten()
    if v.numel() <= n:
        return v
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, v.numel(), (n,), generator=g)
    return v[idx]


def similarity_matrices(small, large, out_patterns, adaln_patterns, sample_n):
    """S_out[i,j] — similarity of output projections, S_in[i,j] — of the other weights."""
    ns, nl = small.num_blocks, large.num_blocks
    rests = sorted(set(small.blocks[0]) & set(large.blocks[0]))
    s_out = torch.zeros(ns, nl)
    s_in = torch.zeros(ns, nl)
    c_out = c_in = 0
    prog = Progress(len(rests), t("Схожесть блоков", "Block similarity"))
    for r in rests:
        prog.step()
        if matches_any(r, adaln_patterns):
            continue
        if len(small.shape(small.blocks[0][r])) < 2:
            continue  # norms/biases are almost the same in all blocks — noise
        is_out = matches_any(r, out_patterns)
        seed = 1234 + zlib.crc32(r.encode()) % 100000
        A = torch.stack([_sample(small.block(i, r), sample_n, seed) for i in range(ns)])
        Bm = torch.stack([_sample(large.block(j, r), sample_n, seed) for j in range(nl)])
        A = A / (A.norm(dim=1, keepdim=True) + 1e-12)
        Bm = Bm / (Bm.norm(dim=1, keepdim=True) + 1e-12)
        S = A @ Bm.T
        if is_out:
            s_out += S
            c_out += 1
        else:
            s_in += S
            c_in += 1
        log(f"  {'[out]' if is_out else '[in] '} {r}")
    if c_out == 0:
        raise SystemExit(t("Не найдено выходных проекций. Укажи шаблоны через --out-patterns "
                           "(посмотри имена слоёв командой info).",
                           "No output projections found. Set patterns with --out-patterns "
                           "(see layer names with the info command)."))
    return s_out / c_out, (s_in / c_in if c_in else s_in), c_out, c_in


def align_monotonic(S):
    """Ordered matching of ns source blocks to nl slots with the maximal sum of S."""
    ns, nl = S.shape
    NEG = float("-inf")
    dp = [[NEG] * (nl + 1) for _ in range(ns + 1)]
    take = [[False] * (nl + 1) for _ in range(ns + 1)]
    for j in range(nl + 1):
        dp[0][j] = 0.0
    for i in range(1, ns + 1):
        for j in range(i, nl + 1):
            skip = dp[i][j - 1]
            use = dp[i - 1][j - 1] + float(S[i - 1, j - 1])
            if use >= skip:
                dp[i][j], take[i][j] = use, True
            else:
                dp[i][j] = skip
    res = [0] * ns
    i, j = ns, nl
    while i > 0:
        if take[i][j]:
            res[i - 1] = j - 1
            i -= 1
        j -= 1
    return res


def build_inserted(small_to_large, nl, copy_choice, s_in=None):
    """For every inserted slot decide which source block it was copied from."""
    large_to_small = {j: i for i, j in enumerate(small_to_large)}
    inserted = []
    for j in range(nl):
        if j in large_to_small:
            continue
        prev = next((large_to_small[x] for x in range(j - 1, -1, -1) if x in large_to_small), None)
        nxt = next((large_to_small[x] for x in range(j + 1, nl) if x in large_to_small), None)
        cands = [c for c in (prev, nxt) if c is not None]
        if copy_choice == "auto" and s_in is not None:
            best = max(cands, key=lambda c: float(s_in[c, j]))
        elif copy_choice == "next":
            best = nxt if nxt is not None else prev
        else:
            best = prev if prev is not None else nxt
        item = {"large": j, "copy_of_small": best, "prev_small": prev, "next_small": nxt}
        if s_in is not None:
            item["score_in"] = round(float(s_in[best, j]), 4)
        inserted.append(item)
    return inserted


def spec_layout(ns, nl, inserted_positions, preset):
    n_ins = nl - ns
    if inserted_positions:
        pos = sorted(set(int(x) for x in inserted_positions.split(",") if x.strip()))
        if len(pos) != n_ins:
            raise SystemExit(t(f"Нужно ровно {n_ins} вставленных позиций (у тебя {len(pos)}).",
                               f"Exactly {n_ins} inserted positions are needed (you gave {len(pos)})."))
        if pos[-1] >= nl or pos[0] < 0:
            raise SystemExit(t(f"Позиции должны быть в диапазоне 0..{nl - 1}.",
                               f"Positions must be within 0..{nl - 1}."))
        return [j for j in range(nl) if j not in set(pos)]
    if preset == "official":
        if (ns, nl) != (28, 40):
            raise SystemExit(t(f"Пресет official только для 28 -> 40 блоков, а тут {ns} -> {nl}.",
                               f"The official preset is only for 28 -> 40 blocks, here it's {ns} -> {nl}."))
        return list(OFFICIAL_SMALL_TO_LARGE)
    if preset == "even":
        # LLaMA-Pro style: the copy goes right after its source block, evenly over the depth.
        after = {round((k + 1) * ns / n_ins) - 1 for k in range(n_ins)}
        if len(after) != n_ins:
            raise SystemExit(t("Пресет even не сработал для этих размеров, задай --inserted вручную.",
                               "The even preset doesn't work for these sizes, set --inserted manually."))
        small_to_large, j = [], 0
        for i in range(ns):
            small_to_large.append(j)
            j += 2 if i in after else 1
        return small_to_large
    raise SystemExit(t("Для spec укажи --inserted 1,4,... (индексы в большой модели) или --preset official/even.",
                       "For spec, give --inserted 1,4,... (indexes in the large model) or --preset official/even."))


def print_map(mp, s_out=None):
    log(t(f"\nКарта ({mp['source']}): {mp['small_blocks']} -> {mp['large_blocks']} блоков",
          f"\nMap ({mp['source']}): {mp['small_blocks']} -> {mp['large_blocks']} blocks"))
    log(t(" малый  ->  большой   схожесть  отрыв", " small  ->  large   similarity  margin"))
    for i, j in enumerate(mp["small_to_large"]):
        line = f"  {i:>3}  ->  {j:>3}"
        if s_out is not None:
            row = s_out[i]
            sc = float(row[j])
            others = torch.cat([row[:j], row[j + 1:]])
            margin = sc - float(others.max())
            flag = t("  <-- неуверенно", "  <-- uncertain") if (sc < 0.3 or margin < 0.05) else ""
            line += f"      {sc:6.3f}   {margin:+6.3f}{flag}"
        log(line)
    log(t("Вставленные слоты (большая модель):", "Inserted slots (large model):"))
    for it in mp["inserted"]:
        extra = (t("  схожесть входных весов ", "  input-weight similarity ") + str(it["score_in"])) if "score_in" in it else ""
        log(t(f"  {it['large']:>3}  копия малого блока {it['copy_of_small']}",
              f"  {it['large']:>3}  copy of small block {it['copy_of_small']}") + extra)


def make_map(small, large, args, source):
    ns, nl = small.num_blocks, large.num_blocks
    if source == "auto":
        log(t(f"\nСчитаю схожесть блоков ({small.label} {ns} vs {large.label} {nl})...",
              f"\nComputing block similarity ({small.label} {ns} vs {large.label} {nl})..."))
        s_out, s_in, c_out, c_in = similarity_matrices(
            small, large, args.out_patterns, args.adaln_patterns, args.sample)
        s2l = align_monotonic(s_out)
        mp = {"source": "auto", "small_blocks": ns, "large_blocks": nl,
              "small_to_large": s2l,
              "inserted": build_inserted(s2l, nl, "auto", s_in if c_in else None),
              "scores": [round(float(s_out[i, j]), 4) for i, j in enumerate(s2l)]}
        return mp, s_out
    s2l = spec_layout(ns, nl, args.inserted, args.preset)
    mp = {"source": "spec", "small_blocks": ns, "large_blocks": nl,
          "small_to_large": s2l,
          "inserted": build_inserted(s2l, nl, args.copy_from)}
    if args.preset == "official" and not args.inserted:
        mp["source"] = "official"
        for it in mp["inserted"]:
            it["copy_of_small"] = OFFICIAL_INSERTED[it["large"]]
    return mp, None


def cmd_map(args):
    a = Model(args.small, "small")
    b = Model(args.large, "large")
    a.describe()
    b.describe()
    if a.num_blocks >= b.num_blocks:
        raise SystemExit(t("--small должна иметь меньше блоков, чем --large.",
                           "--small must have fewer blocks than --large."))
    if args.source == "compare":
        auto, s_out = make_map(a, b, args, "auto")
        spec, _ = make_map(a, b, args, "spec")
        print_map(auto, s_out)
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(auto["small_to_large"], spec["small_to_large"])) if x != y]
        log(t(f"\nСравнение auto vs spec: расхождений {len(diff)} из {a.num_blocks}",
              f"\nauto vs spec: {len(diff)} differences out of {a.num_blocks}"))
        for i, x, y in diff:
            log(t(f"  малый {i}: auto -> {x}, spec -> {y}", f"  small {i}: auto -> {x}, spec -> {y}"))
        mp = auto if args.keep == "auto" else spec
    else:
        mp, s_out = make_map(a, b, args, args.source)
        print_map(mp, s_out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(mp, fh, ensure_ascii=False, indent=2)
        log(t("\nКарта сохранена: ", "\nMap saved: ") + args.out)


# -------------------------------------------------------------- cross -----

def load_or_make_map(args, ft_base, target):
    small_n, large_n = sorted((ft_base.num_blocks, target.num_blocks))
    if args.map_source == "file":
        if not args.map_file:
            raise SystemExit(t("--map-source file требует --map-file.", "--map-source file needs --map-file."))
        with open(args.map_file, encoding="utf-8") as fh:
            mp = json.load(fh)
        print_map(mp)
    else:
        if args.map_source == "auto":
            ref_path = args.ref_base
            if not ref_path:
                log(t("ВНИМАНИЕ: --ref-base не задан, карту считаю по самой --target. "
                      "Лучше указать базу другой архитектуры (точнее).",
                      "WARNING: no --ref-base, computing the map from --target itself. "
                      "A base model of the target's architecture is more accurate."))
                ref = target
            else:
                ref = Model(ref_path, "ref-base")
                ref.describe()
                if ref.num_blocks != target.num_blocks:
                    raise SystemExit(t("--ref-base должна быть той же архитектуры, что и --target.",
                                       "--ref-base must have the same architecture as --target."))
            small, large = (ft_base, ref) if ft_base.num_blocks < ref.num_blocks else (ref, ft_base)
        else:
            small, large = (ft_base, target) if ft_base.num_blocks < target.num_blocks else (target, ft_base)
        mp, s_out = make_map(small, large, args, args.map_source)
        print_map(mp, s_out)
        if args.save_map:
            with open(args.save_map, "w", encoding="utf-8") as fh:
                json.dump(mp, fh, ensure_ascii=False, indent=2)
            log(t("Карта сохранена: ", "Map saved: ") + args.save_map)
    if (mp["small_blocks"], mp["large_blocks"]) != (small_n, large_n):
        raise SystemExit(t(f"Карта рассчитана на {mp['small_blocks']}->{mp['large_blocks']}, "
                           f"а модели {small_n}/{large_n}.",
                           f"The map is for {mp['small_blocks']}->{mp['large_blocks']}, "
                           f"but the models are {small_n}/{large_n}."))
    return mp


def cmd_cross(args):
    T = Model(args.target, "target")
    F = Model(args.ft, "ft")
    FB = Model(args.ft_base, "ft-base")
    for x in (T, F, FB):
        x.describe()
    if F.num_blocks != FB.num_blocks:
        raise SystemExit(t("У --ft и --ft-base разное число блоков: база выбрана неверно.",
                           "--ft and --ft-base have different block counts: wrong base."))
    if F.num_blocks == T.num_blocks:
        raise SystemExit(t("Архитектуры совпадают: используй команду merge (A=target, B=ft, C=ft-base).",
                           "Same architecture: use the merge command (A=target, B=ft, C=ft-base)."))

    mp = load_or_make_map(args, FB, T)
    up = F.num_blocks < T.num_blocks
    s2l = mp["small_to_large"]
    l2s = {j: i for i, j in enumerate(s2l)}
    ins = {it["large"]: it for it in mp["inserted"]}

    # Plan: for every target block — a list of (source block, coefficient).
    plan = defaultdict(list)
    if up:
        for j in range(T.num_blocks):
            if j in l2s:
                plan[j].append((l2s[j], 1.0))
            else:
                it = ins[j]
                if args.m_inserted == 0 or args.inserted_source == "zero":
                    continue
                src = args.inserted_source
                if src == "copy":
                    plan[j].append((it["copy_of_small"], args.m_inserted))
                elif src == "prev" and it["prev_small"] is not None:
                    plan[j].append((it["prev_small"], args.m_inserted))
                elif src == "next" and it["next_small"] is not None:
                    plan[j].append((it["next_small"], args.m_inserted))
                elif src == "avg":
                    srcs = [s for s in (it["prev_small"], it["next_small"]) if s is not None]
                    for s in srcs:
                        plan[j].append((s, args.m_inserted / len(srcs)))
                else:
                    plan[j].append((it["copy_of_small"], args.m_inserted))
    else:
        for i in range(T.num_blocks):
            plan[i].append((s2l[i], 1.0))
        if args.fold_inserted:
            for j, it in ins.items():
                plan[it["copy_of_small"]].append((j, args.fold_inserted))
        else:
            log(t(f"\nДельты {len(ins)} вставленных блоков файнтюна отбрасываются "
                  f"(--fold-inserted, чтобы частично влить их в соседей).",
                  f"\nDeltas of the fine-tune's {len(ins)} inserted blocks are dropped "
                  f"(use --fold-inserted to partly fold them into their neighbours)."))

    log(t("\nНаправление: ", "\nDirection: ") +
        (t("2B -> 2.9B (вверх)", "2B -> 2.9B (up)") if up else t("2.9B -> 2B (вниз)", "2.9B -> 2B (down)")))
    mult = Mults(args)
    warn_rule_overlap(mult.deep, T.blocks[0])
    stats = DeltaStats()
    out = {}
    changed = 0
    prog = Progress(T.num_blocks + 1, t("Перенос", "Porting"))

    for j in range(T.num_blocks):
        for r, kt in T.blocks[j].items():
            tt = T.get(kt)
            acc = None
            m = mult.for_block(j, r)
            if m != 0:
                for (src, coef) in plan.get(j, []):
                    kf = F.blocks.get(src, {}).get(r)
                    kb = FB.blocks.get(src, {}).get(r)
                    if kf is None or kb is None:
                        continue
                    tf, tb = F.get(kf), FB.get(kb)
                    if tf.shape != tt.shape or tb.shape != tt.shape:
                        log(t("  форма не совпадает, пропуск: ", "  shape mismatch, skipped: ") + f"blocks.{j}.{r}")
                        continue
                    tff, tbf = tf.float(), tb.float()
                    if coef == 1.0:
                        stats.add(tff, tbf)
                    d = (tff - tbf) * coef
                    acc = d if acc is None else acc + d
            if acc is not None:
                out[kt] = cast(tt.float() + m * acc, tt, args.save_dtype)
                changed += 1
            else:
                out[kt] = cast(tt, tt, args.save_dtype)
        prog.step()

    # Non-block weights: embedders, final layer, adapter etc. — by name.
    for nk, kt in T.other.items():
        tt = T.get(kt)
        kf, kb = F.other.get(nk), FB.other.get(nk)
        m = mult.for_other(nk)
        if kf is None or kb is None or m == 0:
            out[kt] = cast(tt, tt, args.save_dtype)
            continue
        tf, tb = F.get(kf), FB.get(kb)
        if tf.shape != tt.shape or tb.shape != tt.shape:
            out[kt] = cast(tt, tt, args.save_dtype)
            continue
        out[kt] = cast(tt.float() + m * (tf.float() - tb.float()), tt, args.save_dtype)
        changed += 1
    prog.finish()

    stats.report("ft - ft-base")
    log(t(f"Изменено тензоров: {changed} из {len(out)}", f"Changed tensors: {changed} of {len(out)}"))
    if args.dry_run:
        log(t("dry-run: файл не сохранён.", "dry-run: nothing saved."))
        return
    save(out, args.out, T.metadata, {
        "mode": "cross", "direction": "up" if up else "down",
        "target": args.target, "ft": args.ft, "ft_base": args.ft_base,
        "map_source": mp["source"], "small_to_large": s2l,
        "M": args.m, "m_adaln": mult.m_adaln, "m_nonblock": mult.m_nonblock,
        "m_inserted": args.m_inserted, "inserted_source": args.inserted_source,
        "fold_inserted": args.fold_inserted, "block_m": args.block_m, "deep": args.deep})


# ---------------------------------------------------------------- info ----

def cmd_info(args):
    M = Model(args.model, "model")
    M.describe()
    rests = sorted(M.blocks[0])
    log(t(f"\nСлои внутри блока 0 ({len(rests)}):", f"\nLayers inside block 0 ({len(rests)}):"))
    for r in rests:
        tag = []
        if matches_any(r, args.out_patterns):
            tag.append("out")
        if matches_any(r, args.adaln_patterns):
            tag.append("adaln")
        log(f"  {r:60s} {str(M.shape(M.blocks[0][r])):22s} {','.join(tag)}")
    groups = Counter(k.split(".")[0] for k in M.other)
    log(t(f"\nНе-блочные веса ({len(M.other)}): ", f"\nNon-block weights ({len(M.other)}): ")
        + ", ".join(f"{g}({n})" for g, n in groups.most_common()))
    adapter = sorted(k for k in M.other if k.startswith("llm_adapter"))
    if adapter:
        sub = Counter(".".join(k.split(".")[1:3]) for k in adapter)
        log(t("  llm_adapter (для правил adapter:...): ", "  llm_adapter (for adapter:... rules): ")
            + ", ".join(f"{g}({n})" for g, n in sub.most_common(12)))


# ----------------------------------------------------------------- CLI ----

def add_mult_args(p):
    g = p.add_argument_group(t("множители", "multipliers"))
    g.add_argument("--m", type=float, default=1.0, help=t("общий множитель M (по умолчанию 1.0)",
                                                          "global multiplier M (default 1.0)"))
    g.add_argument("--m-adaln", type=float, default=None,
                   help=t("множитель для слоёв модуляции adaLN (по умолчанию = M; 0 = оставить от target/A)",
                          "multiplier for adaLN modulation layers (default = M; 0 = keep target/A)"))
    g.add_argument("--m-nonblock", type=float, default=None,
                   help=t("множитель для не-блочных весов: эмбеддеры, финальный слой, адаптер (по умолчанию = M)",
                          "multiplier for non-block weights: embedders, final layer, adapter (default = M)"))
    g.add_argument("--deep", default="",
                   help=t("точные правила по блокам и слоям: '0, L16-L27:mlp.layer1 mlp.layer2:0.5, "
                          "L00-L27:adaln:0, adapter:*:0'",
                          "exact rules per block and layer: '0, L16-L27:mlp.layer1 mlp.layer2:0.5, "
                          "L00-L27:adaln:0, adapter:*:0'"))
    g.add_argument("--block-m", default="",
                   help=t("поблочные коэффициенты (умножаются на M) по индексам целевой модели, напр. '0-9:0.5,30-39:1.2'",
                          "per-block factors (times M) by target block index, e.g. '0-9:0.5,30-39:1.2'"))


def add_common(p):
    p.add_argument("--out-patterns", nargs="+", default=DEFAULT_OUT_PATTERNS,
                   help=t("подстроки имён выходных проекций", "substrings of output projection names"))
    p.add_argument("--adaln-patterns", nargs="+", default=DEFAULT_ADALN_PATTERNS,
                   help=t("подстроки имён слоёв модуляции", "substrings of modulation layer names"))


def add_map_args(p):
    g = p.add_argument_group(t("карта блоков", "block map"))
    g.add_argument("--inserted", default="",
                   help=t("spec: индексы вставленных слоёв в 40-слойной модели через запятую (0-based)",
                          "spec: comma-separated indexes of inserted layers in the 40-layer model (0-based)"))
    g.add_argument("--preset", choices=["official", "even"], default=None,
                   help=t("spec: official = опубликованная схема расширения 28->40; even = равномерная догадка",
                          "spec: official = the published 28->40 expansion; even = an even-spacing guess"))
    g.add_argument("--copy-from", choices=["prev", "next"], default="prev",
                   help=t("spec: копией какого соседа считать вставленный слой",
                          "spec: which neighbour an inserted layer is a copy of"))
    g.add_argument("--sample", type=int, default=1_000_000,
                   help=t("auto: сколько элементов тензора брать для оценки схожести",
                          "auto: tensor elements to sample for similarity"))


def main():
    safe_print_setup()
    ap = argparse.ArgumentParser(description="Anima Merger 2B / 2.9B",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("merge", help=t("A + (B - C) * M, одна архитектура (в т.ч. замена базы)",
                                       "A + (B - C) * M, same architecture (incl. rebase)"))
    p.add_argument("--a", required=True, help=t("модель, К КОТОРОЙ добавляем (для rebase: новая база)",
                                                "model to add TO (for rebase: the new base)"))
    p.add_argument("--b", required=True, help=t("модель, ИЗ КОТОРОЙ берём дельту (файнтюн)",
                                                "model to take the delta FROM (fine-tune)"))
    p.add_argument("--c", required=True, help=t("база, от которой обучалась B", "the base B was trained from"))
    p.add_argument("--out", required=False, help=t("выходной .safetensors", "output .safetensors"))
    p.add_argument("--save-dtype", choices=list(DTYPES), default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mode", choices=MODES, default="add",
                   help=t("add = обычная дельта; topk = только самые сильные изменения; dare = случайное "
                          "прореживание; ortho = без части, параллельной весам A (не меняет масштаб A); "
                          "ortho_delta = без части, параллельной изменению A от C (не даёт сложить одно и то же "
                          "дважды); normdir = без «выцветания»",
                          "add = plain delta; topk = only the strongest changes; dare = random drop; "
                          "ortho = without the part parallel to A's weights (keeps A's scale); "
                          "ortho_delta = without the part parallel to A's own change from C (no double dose); "
                          "normdir = no 'fading'"))
    p.add_argument("--density", type=float, default=0.2, help=t("topk/dare: доля дельты, которую оставить",
                                                                "topk/dare: share of the delta to keep"))
    p.add_argument("--seed", type=int, default=0, help=t("dare: сид", "dare: seed"))
    p.add_argument("--proj", type=float, default=1.0,
                   help=t("ortho_delta: какую долю совпадающей с A части убрать (1 = всю, 0 = как add)",
                          "ortho_delta: share of the part matching A to remove (1 = all, 0 = same as add)"))
    add_mult_args(p)
    add_common(p)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("map", help=t("построить карту блоков 2B <-> 2.9B", "build a 2B <-> 2.9B block map"))
    p.add_argument("--small", required=True, help=t("база 2B (28 блоков)", "2B base (28 blocks)"))
    p.add_argument("--large", required=True, help=t("база 2.9B (40 блоков)", "2.9B base (40 blocks)"))
    p.add_argument("--source", choices=["auto", "spec", "compare"], default="auto")
    p.add_argument("--keep", choices=["auto", "spec"], default="auto",
                   help=t("compare: какую карту сохранить в --out", "compare: which map to save to --out"))
    p.add_argument("--out", default=None, help=t("куда сохранить карту (.json)", "where to save the map (.json)"))
    add_map_args(p)
    add_common(p)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("cross", help=t("перенести файнтюн между 2B и 2.9B", "port a fine-tune between 2B and 2.9B"))
    p.add_argument("--target", required=True, help=t("модель другой архитектуры, к которой добавляем",
                                                     "model of the other architecture to add to"))
    p.add_argument("--ft", required=True, help=t("файнтюн, который переносим", "the fine-tune to port"))
    p.add_argument("--ft-base", required=True, help=t("база, от которой обучался --ft", "the base --ft was trained from"))
    p.add_argument("--out", required=False)
    p.add_argument("--map-source", choices=["auto", "spec", "file"], default="auto",
                   help=t("откуда брать карту блоков", "where the block map comes from"))
    p.add_argument("--map-file", default=None, help=t("file: путь к карте из команды map", "file: map from the map command"))
    p.add_argument("--ref-base", default=None,
                   help=t("auto: база той же архитектуры, что target (точнее, чем считать по самой target)",
                          "auto: a base of the target's architecture (more accurate than the target itself)"))
    p.add_argument("--save-map", default=None, help=t("сохранить использованную карту в .json", "save the used map to .json"))
    p.add_argument("--m-inserted", type=float, default=0.0,
                   help=t("вверх: доля дельты для вставленных слоёв (0 = не трогать, 0.2–0.5 = осторожно)",
                          "up: delta share for inserted layers (0 = leave alone, 0.2–0.5 = careful)"))
    p.add_argument("--inserted-source", choices=["copy", "prev", "next", "avg", "zero"], default="copy",
                   help=t("вверх: чью дельту давать вставленным слоям", "up: whose delta the inserted layers get"))
    p.add_argument("--fold-inserted", type=float, default=0.0,
                   help=t("вниз: влить дельты вставленных слоёв в их оригиналы с этим коэффициентом",
                          "down: fold inserted-layer deltas into their originals with this factor"))
    p.add_argument("--save-dtype", choices=list(DTYPES), default=None)
    p.add_argument("--dry-run", action="store_true")
    add_mult_args(p)
    add_map_args(p)
    add_common(p)
    p.set_defaults(func=cmd_cross)

    p = sub.add_parser("info", help=t("показать структуру модели", "show model structure"))
    p.add_argument("--model", required=True)
    add_common(p)
    p.set_defaults(func=cmd_info)

    import lora_tools
    lora_tools.add_parsers(sub)
    import analysis
    analysis.add_parsers(sub)

    args = ap.parse_args()
    if args.cmd in ("merge", "cross", "bake", "extract", "lora-merge") and not args.dry_run and not args.out:
        ap.error(t("укажи --out (или --dry-run)", "give --out (or --dry-run)"))
    with torch.no_grad():
        args.func(args)


if __name__ == "__main__":
    main()
