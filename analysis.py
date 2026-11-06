# -*- coding: utf-8 -*-
"""Read-only analyzers for Anima models and LoRAs. Nothing is ever written next to the models:
reports go to the reports/ folder as .html (charts), .txt and .json.

  parent    — which model a fine-tune / merge was made from
  family    — family tree: distances, nearest relatives, identical tensors
  genome    — shared "genes" of a family + what each model is assembled from (merge decomposition)
  blocks    — where a model differs from a reference: per block, per layer type, inserted 2.9B blocks
  meta      — recipes and metadata from file headers (AnimaMerger steps, merge recipes, LoRA training)
  lora-info — LoRA analysis; several checkpoints of one training = a curve over epochs/steps
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from amutil import (t, log, Progress, pick_device, PREFIXES, short, unique_labels, ltype, read_header, file_kind)
from report import Report
import profiles

FLOAT = {"BF16", "F16", "F32", "F64"}
BLK = re.compile(r"^blocks\.(\d+)\.(.+)$")
OFFICIAL_INSERTED = {2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36}
OFFICIAL_SMALL_TO_LARGE = [j for j in range(40) if j not in OFFICIAL_INSERTED]
TYPES = ["self_attn", "cross_attn", "mlp", "adaln", "other", "adapter", "nonblock"]


def type_names():
    return {"self_attn": "self_attn", "cross_attn": "cross_attn", "mlp": "mlp", "adaln": "adaLN",
            "other": t("прочее в блоке", "other in block"), "adapter": "llm_adapter",
            "nonblock": t("вне блоков", "outside blocks")}


# ------------------------------------------------------------------ reading --

class Src:
    """Lazy reader with normalized keys (prefixes net./diffusion_model./... removed)."""

    def __init__(self, path):
        self.path = str(path)
        self.f = safe_open(self.path, framework="pt", device="cpu")
        self.meta = self.f.metadata() or {}
        self.map = {}
        self.prefixes = set()
        for k in self.f.keys():
            nk = k
            for p in PREFIXES:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    self.prefixes.add(p)
                    break
            self.map[nk] = k
        nbs = {int(m.group(1)) for k in self.map for m in [BLK.match(k)] if m}
        self.nblocks = (max(nbs) + 1) if nbs else 0

    def has(self, k):
        return k in self.map

    def shape(self, k):
        return tuple(self.f.get_slice(self.map[k]).get_shape())

    def dtype(self, k):
        return self.f.get_slice(self.map[k]).get_dtype()

    def get(self, k, dev):
        return self.f.get_tensor(self.map[k]).to(dev).float().reshape(-1)


def open_src(path, exact=False):
    """Full reader (exact) or a profile sketch (fast, cached)."""
    return Src(path) if exact else profiles.Sketch(path)


def mode_note(R, exact):
    if not exact:
        R.note(t(f"Посчитано по профилям моделей (выборка {profiles.SKETCH} весов из каждого тензора, "
                 f"погрешность расстояний и долей около 1–2%; одинаковые тензоры — точно, по хешу). "
                 f"Для точного расчёта включи «точно, без профилей».",
                 f"Computed from model profiles (a {profiles.SKETCH}-weight sample of every tensor; distances and "
                 f"shares are within about 1–2%; identical tensors are exact, by hash). "
                 f"Turn on 'exact, no profiles' for an exact run."))


def key_type(k):
    if "llm_adapter" in k:
        return None, "adapter"
    m = BLK.match(k)
    if not m:
        return None, "nonblock"
    return int(m.group(1)), ltype(m.group(2))


_FPW = {}


def fingerprint(x):
    """Cheap fingerprint to detect bit-identical tensors without pairwise comparisons."""
    n = x.numel()
    key = (n, str(x.device))
    w = _FPW.get(key)
    if w is None:
        w = torch.cos(torch.arange(n, device=x.device, dtype=torch.float32) * 0.7071)
        if len(_FPW) > 64:
            _FPW.clear()
        _FPW[key] = w
    return torch.stack([x.sum(), (x * x).sum(), (x * w).sum()])


def gram(Y):
    if Y.shape[1] <= 4_000_000:
        return (Y @ Y.T).double()
    g = None
    for j in range(0, Y.shape[1], 4_000_000):
        c = Y[:, j:j + 4_000_000]
        p = (c @ c.T).double()
        g = p if g is None else g + p
    return g


def common_keys(ref, models):
    keys, dropped = [], 0
    for k in ref.map:
        if ref.dtype(k) not in FLOAT:
            continue
        shp = ref.shape(k)
        if all(m.has(k) and m.shape(k) == shp for m in models):
            keys.append(k)
        else:
            dropped += 1
    return keys, dropped


def accumulate(models, ref, dev, label):
    """One pass over all tensors. Deltas Y_i = X_i - ref. Returns Gram matrices (whole model,
    per block, per layer type), squared norms of ref, and counts of bit-identical tensors."""
    n = len(models)
    keys, dropped = common_keys(ref, models)
    if dropped:
        log(t(f"  тензоров пропущено (нет у какой-то модели или другая форма): {dropped}",
              f"  tensors skipped (missing in some model or different shape): {dropped}"))
    nb = max([m.nblocks for m in models] + [ref.nblocks])
    G = torch.zeros(n, n, dtype=torch.float64, device=dev)
    GB = torch.zeros(max(nb, 1), n, n, dtype=torch.float64, device=dev)
    GT = {tp: torch.zeros(n, n, dtype=torch.float64, device=dev) for tp in TYPES}
    BS = {"all": 0.0, "blk": np.zeros(max(nb, 1)), "type": {tp: 0.0 for tp in TYPES}}
    SAME = torch.zeros(n, n, dtype=torch.int64, device=dev)
    prog = Progress(len(keys), label)
    for k in keys:
        r = ref.get(k, dev)
        X = torch.stack([m.get(k, dev) if m is not ref else r for m in models])
        if all(getattr(m, "sketch", False) for m in models):
            hs = [m.fp(k) for m in models]
            SAME += torch.tensor([[int(a == b and a is not None) for b in hs] for a in hs], device=dev)
        else:
            fp = torch.stack([fingerprint(x) for x in X])
            SAME += (fp[:, None, :] == fp[None, :, :]).all(-1).long()
        X -= r
        g = gram(X)
        del X
        bi, tp = key_type(k)
        G += g
        GT[tp] += g
        bs = float((r.double() * r.double()).sum()) if r.numel() < 50_000_000 else float((r * r).sum())
        BS["all"] += bs
        BS["type"][tp] += bs
        if bi is not None:
            GB[bi] += g
            BS["blk"][bi] += bs
        prog.step()
    return {"G": G.cpu().numpy(), "GB": GB.cpu().numpy(), "GT": {k: v.cpu().numpy() for k, v in GT.items()},
            "BS": BS, "SAME": SAME.cpu().numpy(), "nblocks": nb, "nkeys": len(keys)}


def inserted_for(nb):
    return OFFICIAL_INSERTED if nb == 40 else set()


def block_labels(nb):
    ins = inserted_for(nb)
    return [f"{b}*" if b in ins else str(b) for b in range(nb)]


def hl_legend():
    return (t("старые блоки", "original blocks"), t("вставленные блоки 2.9B", "inserted 2.9B blocks"))


def resolve_models(args_models):
    """Files as given; folders are searched recursively (subfolders included). Duplicates are dropped."""
    out, seen = [], set()
    for p in args_models or []:
        items = sorted((str(x) for x in Path(p).rglob("*.safetensors")), key=str.lower) if os.path.isdir(p) else [p]
        for x in items:
            key = os.path.normcase(os.path.abspath(x))
            if key not in seen:
                seen.add(key)
                out.append(x)
    return out


# ------------------------------------------------------------------- parent --

def cmd_parent(args):
    dev = pick_device(args.device)
    ex = getattr(args, "exact", False)
    T = open_src(args.target, ex)
    cands = [c for c in resolve_models(args.cands) if os.path.abspath(c) != os.path.abspath(args.target)]
    C = []
    for p in cands:
        if file_kind(p)["kind"] != "model":
            continue
        s = open_src(p, ex)
        if s.nblocks != T.nblocks:
            log(t(f"  пропуск {short(p)}: {s.nblocks} блоков, а у цели {T.nblocks}",
                  f"  skipped {short(p)}: {s.nblocks} blocks, the target has {T.nblocks}"))
            continue
        C.append(s)
    B = open_src(args.base, ex) if args.base else None
    if B is not None and all(os.path.abspath(c.path) != os.path.abspath(B.path) for c in C) \
            and B.nblocks == T.nblocks:
        C.append(B)
    if not C:
        raise SystemExit(t("Нет кандидатов с тем же числом блоков.", "No candidates with the same block count."))
    names = unique_labels([c.path for c in C], 34)
    log(t("Цель: ", "Target: ") + short(T.path, 60))
    log(t("Кандидаты: ", "Candidates: ") + ", ".join(names))
    nc, nb = len(C), max(T.nblocks, 1)
    d2 = torch.zeros(nc, dtype=torch.float64, device=dev)
    d2b = torch.zeros(nc, nb, dtype=torch.float64, device=dev)
    d2t = torch.zeros(nc, len(TYPES), dtype=torch.float64, device=dev)
    tb2 = torch.zeros((), dtype=torch.float64, device=dev)
    cb2 = torch.zeros(nc, dtype=torch.float64, device=dev)
    dot = torch.zeros(nc, dtype=torch.float64, device=dev)
    keys = [k for k in T.map if T.dtype(k) in FLOAT]
    prog = Progress(len(keys), t("Поиск родителя", "Finding parent"))
    for k in keys:
        prog.step()
        x = T.get(k, dev)
        shp = T.shape(k)
        bi, tp = key_type(k)
        ti = TYPES.index(tp)
        bb = None
        if B is not None and B.has(k) and B.shape(k) == shp:
            bb = B.get(k, dev)
            tdel = x - bb
            tb2 += (tdel * tdel).sum().double()
        for j, c in enumerate(C):
            if not c.has(k) or c.shape(k) != shp:
                continue
            y = c.get(k, dev)
            d = x - y
            s = (d * d).sum().double()
            d2[j] += s
            d2t[j, ti] += s
            if bi is not None:
                d2b[j, bi] += s
            if bb is not None:
                cd = y - bb
                cb2[j] += (cd * cd).sum().double()
                dot[j] += (tdel * cd).sum().double()
    d2, d2b, d2t = d2.cpu().numpy(), d2b.cpu().numpy(), d2t.cpu().numpy()
    tb2, cb2, dot = float(tb2), cb2.cpu().numpy(), dot.cpu().numpy()
    order = list(np.argsort(d2))

    R = Report("parent", t("Поиск родителя: ", "Parent search: ") + short(T.path, 60),
               t(f"База для масштаба: {short(B.path, 60) if B else 'не задана'}. Только чтение.",
                 f"Reference base: {short(B.path, 60) if B else 'none'}. Read-only."))
    R.data.update({"target": T.path, "base": B.path if B else None, "candidates": [c.path for c in C], "exact": ex})
    mode_note(R, ex)
    R.h2(t("1. Кто ближе всего", "1. Who is closest"))
    R.note(t("ratio = |цель − кандидат| / |цель − база|: чем меньше, тем вероятнее родитель. "
             "cos — насколько изменения цели и кандидата (от базы) смотрят в одну сторону.",
             "ratio = |target − candidate| / |target − base|: smaller = more likely the parent. "
             "cos — how much the target's and the candidate's changes (from the base) point the same way."))
    rows = []
    ref = math.sqrt(tb2) if tb2 > 0 else math.sqrt(max(d2.max(), 1e-30))
    for r, j in enumerate(order):
        ratio = math.sqrt(d2[j]) / ref
        cos = dot[j] / math.sqrt(tb2 * cb2[j]) if tb2 > 0 and cb2[j] > 0 else None
        rows.append([names[j], ratio, cos, t("← вероятный родитель", "← likely parent") if r == 0 else ""])
    R.table([t("кандидат", "candidate"), "ratio", "cos", ""], rows, best_row=0)
    R.bars([names[j] for j in order], [math.sqrt(d2[j]) / ref for j in order],
           t("Расстояние до цели (меньше = ближе)", "Distance to the target (smaller = closer)"), horizontal=True)
    R.data["ranking"] = [{"candidate": C[j].path, "ratio": math.sqrt(d2[j]) / ref} for j in order]
    if len(order) > 1:
        g = math.sqrt(d2[order[1]] / max(d2[order[0]], 1e-30))
        R.data["gap"] = g
        if g > 1.3:
            R.badge("good", t(f"Уверенно: лучший кандидат ближе второго в {g:.2f} раза.",
                              f"Confident: the best candidate is {g:.2f}× closer than the second."))
        elif g > 1.1:
            R.badge("warn", t(f"Отрыв небольшой ({g:.2f}×): родитель вероятен, но проверь по блокам ниже.",
                              f"Small gap ({g:.2f}×): a likely parent, but check the per-block view below."))
        else:
            R.badge("bad", t(f"Кандидаты почти равны ({g:.2f}×): точного родителя в списке, похоже, нет.",
                             f"Candidates are nearly equal ({g:.2f}×): the exact parent is probably not in the list."))
    if T.nblocks:
        top = order[:3]
        R.h2(t("2. По блокам (3 лучших кандидата)", "2. Per block (top 3 candidates)"))
        R.note(t("Расстояние по блоку, нормировано на худшего кандидата. Блок с маленьким числом у кандидата = "
                 "этот блок взят у него (поблочный мердж виден как «полосы»).",
                 "Per-block distance, normalized by the worst candidate. A small number = this block was taken "
                 "from that candidate (block-wise merges show up as 'stripes')."))
        worst = order[-1]
        M = [[math.sqrt(d2b[j, b] / d2b[worst, b]) if d2b[worst, b] > 0 else 0.0 for b in range(T.nblocks)]
             for j in top]
        R.heatmap([names[j] for j in top], block_labels(T.nblocks), M,
                  t("ratio по блокам (светлее = ближе)", "per-block ratio (lighter = closer)"), reverse=True,
                  show_values=T.nblocks <= 28, vmin=0, vmax=1)
        best = order[0]
        R.h2(t(f"3. Куда легло обучение: цель − {names[best]}", f"3. Where the training went: target − {names[best]}"))
        tot = max(d2[best], 1e-30)
        R.bars(block_labels(T.nblocks), [100 * d2b[best, b] / tot for b in range(T.nblocks)],
               t("Доля изменений по блокам, %", "Share of changes per block, %"), fmt_s="{:.1f}",
               highlight=inserted_for(T.nblocks), hl_label=hl_legend())
        tn = type_names()
        R.table([t("тип слоёв", "layer type"), "%"],
                [[tn[tp], 100 * d2t[best, i] / tot] for i, tp in enumerate(TYPES) if d2t[best, i] > 0],
                fmts={1: "{:.1f}"})
    R.save("parent")


# ------------------------------------------------------------------- family --

def load_models(paths, need_blocks=None, exact=False):
    S = []
    for p in paths:
        if file_kind(p)["kind"] != "model":
            log(t(f"  пропуск {short(p)}: это не модель (LoRA?)", f"  skipped {short(p)}: not a model (LoRA?)"))
            continue
        s = open_src(p, exact)
        if need_blocks and s.nblocks != need_blocks:
            log(t(f"  пропуск {short(p)}: {s.nblocks} блоков вместо {need_blocks}",
                  f"  skipped {short(p)}: {s.nblocks} blocks instead of {need_blocks}"))
            continue
        S.append(s)
    return S


def cmd_family(args):
    dev = pick_device(args.device)
    ex = getattr(args, "exact", False)
    ref = open_src(args.ref, ex)
    paths = [p for p in resolve_models(args.models) if os.path.abspath(p) != os.path.abspath(args.ref)]
    models = [ref] + load_models(paths, ref.nblocks, ex)
    if len(models) < 3:
        raise SystemExit(t("Нужно хотя бы 2 модели кроме эталона.", "Need at least 2 models besides the reference."))
    names = unique_labels([m.path for m in models], 22)
    n = len(models)
    A = accumulate(models, ref, dev, t("Семья", "Family"))
    G, GB, SAME = A["G"], A["GB"], A["SAME"]
    dg = np.diag(G)
    D2 = np.maximum(dg[:, None] + dg[None, :] - 2 * G, 0)
    nb = A["nblocks"]
    R = Report("family", t("Семейное древо", "Family tree"),
               t(f"Эталон: {names[0]}. Моделей: {n}, сравнено тензоров: {A['nkeys']}. Только чтение.",
                 f"Reference: {names[0]}. Models: {n}, tensors compared: {A['nkeys']}. Read-only."))
    R.data.update({"models": [m.path for m in models], "dist2": D2.tolist(), "same": SAME.tolist(), "exact": ex})
    mode_note(R, ex)
    refmax = math.sqrt(max(dg.max(), 1e-30))
    R.h2(t("1. Насколько каждая модель ушла от эталона", "1. How far each model is from the reference"))
    R.bars(names[1:], [math.sqrt(dg[i]) / refmax for i in range(1, n)],
           t("|модель − эталон|, относительно самой далёкой", "|model − reference|, relative to the farthest"),
           horizontal=True)
    R.table([t("модель", "model"), t("расстояние", "distance"), t("тензоров бит в бит как у эталона", "tensors identical to the reference")],
            [[names[i], math.sqrt(dg[i]) / refmax, int(SAME[i, 0])] for i in range(1, n)])
    R.h2(t("2. Матрица расстояний", "2. Distance matrix"))
    R.note(t("Темнее = ближе. 0 — одна и та же модель.", "Darker = closer. 0 = the same model."))
    R.heatmap(names, names, [[math.sqrt(D2[i, j]) / refmax for j in range(n)] for i in range(n)],
              t("расстояние (доля от самой далёкой модели)", "distance (share of the farthest model)"), reverse=True,
              vmin=0)
    R.h2(t("3. Ближайшие родственники", "3. Nearest relatives"))
    R.note(t("ratio = расстояние до соседа / расстояние модели от эталона. < 0.7 — явное родство. "
             "Много бит-в-бит одинаковых тензоров — почти наверняка прямой родитель или потомок.",
             "ratio = distance to the neighbour / the model's distance from the reference. < 0.7 — clearly related. "
             "Many bit-identical tensors — almost certainly a direct parent or child."))
    rows = []
    for i in range(1, n):
        own = max(dg[i], 1e-30)
        cand = sorted((D2[i, j] / own, j) for j in range(n) if j != i)
        r1, j1 = cand[0]
        zb = [b for b in range(nb) if GB[b, i, i] + GB[b, j1, j1] - 2 * GB[b, i, j1] <= 1e-9 * (GB[b, i, i] + GB[b, j1, j1] + 1e-30)]
        others = [j for j in range(n) if j != i]
        js = max(others, key=lambda j: SAME[i, j])
        if SAME[i, js] > 0:
            zs = [b for b in range(nb) if GB[b, i, i] + GB[b, js, js] - 2 * GB[b, i, js]
                  <= 1e-9 * (GB[b, i, i] + GB[b, js, js] + 1e-30)]
            same_txt = f"{names[js]} ({int(SAME[i, js])}" + (
                t(f"; блоки {', '.join(map(str, zs[:14]))}{'…' if len(zs) > 14 else ''})",
                  f"; blocks {', '.join(map(str, zs[:14]))}{'…' if len(zs) > 14 else ''})") if zs else ")")
        else:
            same_txt = "—"
        rows.append([names[i], names[j1], math.sqrt(r1), int(SAME[i, j1]),
                     f"{names[cand[1][1]]} ({math.sqrt(cand[1][0]):.2f})" if len(cand) > 1 else "",
                     (", ".join(map(str, zb[:12])) + ("…" if len(zb) > 12 else "")) if zb else "—", same_txt])
    R.table([t("модель", "model"), t("ближайший", "nearest"), "ratio", t("одинаковых тензоров", "identical tensors"),
             t("второй", "second"), t("блоки, совпадающие полностью", "fully identical blocks"),
             t("больше всего одинаковых тензоров с", "most identical tensors with")], rows)
    R.save("family")


# ------------------------------------------------------------------- genome --

def nnls_gram(A, y, iters=3000):
    """min |Σ w_i d_i − d_t|², w ≥ 0, via the Gram matrix (coordinate descent)."""
    w = np.zeros(len(y))
    for _ in range(iters):
        prev = w.copy()
        for i in range(len(y)):
            if A[i, i] <= 0:
                continue
            w[i] = max(0.0, (y[i] - A[i] @ w + A[i, i] * w[i]) / A[i, i])
        if np.abs(w - prev).max() < 1e-10:
            break
    return w


def r2_of(G, t_, idx, w):
    tt = G[t_, t_]
    if tt <= 0 or not idx:
        return 0.0
    return float((2 * w @ G[idx, t_] - w @ G[np.ix_(idx, idx)] @ w) / tt)


def decompose(G, t_, pool, min_gain=0.01):
    """Minimal set of 'ancestors' with non-negative weights (greedy selection + NNLS)."""
    pool = [j for j in pool if j != t_]
    chosen, best_w, best_r2 = [], np.zeros(0), 0.0
    while True:
        cand = None
        for j in pool:
            if j in chosen:
                continue
            idx = chosen + [j]
            w = nnls_gram(G[np.ix_(idx, idx)], G[idx, t_])
            r2 = r2_of(G, t_, idx, w)
            if cand is None or r2 > cand[2]:
                cand = (idx, w, r2)
        if cand is None or cand[2] - best_r2 < min_gain:
            break
        chosen, best_w, best_r2 = cand
        if best_r2 > 0.995:
            break
    keep = [(j, float(x)) for j, x in zip(chosen, best_w) if x > 1e-4]
    return keep, best_r2


def cmd_genome(args):
    dev = pick_device(args.device)
    ex = getattr(args, "exact", False)
    base = open_src(args.base, ex)
    paths = [p for p in resolve_models(args.models) if os.path.abspath(p) != os.path.abspath(args.base)]
    models = load_models(paths, base.nblocks, ex)
    if len(models) < 2:
        raise SystemExit(t("Нужно хотя бы 2 модели кроме базы.", "Need at least 2 models besides the base."))
    n = len(models)
    names = unique_labels([m.path for m in models], 22)
    A = accumulate(models, base, dev, t("Геном", "Genome"))
    G, GB, GT, BS, SAME, nb = A["G"], A["GB"], A["GT"], A["BS"], A["SAME"], A["nblocks"]
    dg = np.maximum(np.diag(G), 1e-30)
    C = G / np.sqrt(np.outer(dg, dg))
    R = Report("genome", t("Геном семейства и состав мерджей", "Family genome and merge composition"),
               t(f"База: {short(base.path, 60)}. Моделей: {n}. Каждая модель = база + её изменения (дельта); "
                 f"дельты раскладываются на общие «гены». Только чтение.",
                 f"Base: {short(base.path, 60)}. Models: {n}. Each model = base + its changes (delta); "
                 f"the deltas are split into shared 'genes'. Read-only."))
    R.data.update({"base": base.path, "models": [m.path for m in models], "gram": G.tolist(), "exact": ex})
    mode_note(R, ex)

    R.h2(t("1. Насколько каждая модель отличается от базы", "1. How much each model differs from the base"))
    R.bars(names, [math.sqrt(dg[i] / max(BS["all"], 1e-30)) for i in range(n)],
           "|delta| / |base|", fmt_s="{:.4f}", horizontal=True)

    R.h2(t("2. Похожесть изменений", "2. Similarity of changes"))
    R.note(t("Косинус дельт: 1 = одно и то же изменение, 0 = независимые, < 0 = в разные стороны.",
             "Cosine of deltas: 1 = the same change, 0 = independent, < 0 = opposite."))
    R.heatmap(names, names, C.tolist(), t("косинус дельт", "cosine of deltas"), vmin=min(0.0, float(C.min())), vmax=1)

    R.h2(t("3. Гены — главные общие направления изменений", "3. Genes — the main shared directions of change"))
    R.note(t("Доля = сколько всей изменчивости семейства объясняет ген. У кого он есть: доля изменений модели, "
             "которую объясняет ген, и знак. Ген у всех с одним знаком — общий стиль семейства; "
             "с «+» у одних и «−» у других — развилка веток.",
             "Share = how much of the family's variation the gene explains. Who has it: the share of the model's "
             "changes the gene explains, and its sign. Same sign everywhere = shared family style; "
             "'+' for some and '−' for others = a branch split."))
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    tot = max(w.sum(), 1e-30)
    gene_rows, genes = [], []
    for c in range(min(n, 6)):
        share = w[c] / tot
        if share < 0.02:
            break
        coef = V[:, c] * math.sqrt(max(w[c], 0))
        part = coef ** 2 / dg
        sgn = np.sign(coef)
        if (sgn < 0).sum() > (sgn > 0).sum():
            sgn = -sgn
        items = sorted(range(n), key=lambda i: -part[i])
        desc = ", ".join(f"{names[i]} {'+' if sgn[i] > 0 else '−'}{100 * part[i]:.0f}%" for i in items if part[i] >= 0.05)
        gene_rows.append([t(f"ген {c + 1}", f"gene {c + 1}"), 100 * share, desc])
        genes.append(c)
    R.table([t("ген", "gene"), t("доля, %", "share, %"), t("у кого (сила, знак)", "who has it (strength, sign)")],
            gene_rows, fmts={1: "{:.1f}"})
    R.data["genes"] = [{"share": float(w[c] / tot), "vector": V[:, c].tolist()} for c in genes]

    R.h2(t("4. Из чего собрана каждая модель", "4. What each model is assembled from"))
    R.note(t("Минимальный набор «предков» из этого списка с весами: база + Σ вес × (предок − база). "
             "R² — доля изменений, которую объясняют предки. R² < 0.3 — модель ни из чего тут не собрана "
             "(своё обучение); 0.3–0.9 — есть что-то ещё (другая модель, LoRA, нелинейный мердж); > 0.9 — собрана из них.",
             "The minimal set of 'ancestors' from this list with weights: base + Σ weight × (ancestor − base). "
             "R² — the share of changes the ancestors explain. R² < 0.3 — not assembled from anything here "
             "(own training); 0.3–0.9 — something else too (another model, a LoRA, a non-linear merge); > 0.9 — assembled from them."))
    rows, decs, bdecs = [], {}, {}
    for i in range(n):
        keep, r2 = decompose(G, i, list(range(n)))
        decs[i] = (keep, r2)
        # Per-block decomposition: catches block-weighted merges (MBW) that one global weight can't explain
        br2, per_block = None, {}
        if nb:
            num = den = 0.0
            for b in range(nb):
                if GB[b, i, i] <= 0:
                    continue
                kb, rb = decompose(GB[b], i, list(range(n)), min_gain=0.02)
                per_block[b] = dict(kb)
                num += rb * GB[b, i, i]
                den += GB[b, i, i]
            br2 = num / den if den > 0 else None
        bdecs[i] = (br2, per_block)
        if br2 is not None and br2 >= 0.9 and br2 - r2 > 0.15:
            verdict = t("поблочный мердж", "block-weighted merge")
        elif r2 < 0.3:
            verdict = t("своё обучение", "own training")
        elif r2 < 0.9:
            verdict = t("частично", "partly")
        else:
            verdict = t("собрана из них", "assembled from them")
        rows.append([names[i], r2, br2,
                     ", ".join(f"{x:.2f}×{names[j]}" for j, x in sorted(keep, key=lambda z: -z[1])) or "—", verdict])
    R.table([t("модель", "model"), "R²", t("R² по блокам", "R² per block"),
             t("предки (вес × модель)", "ancestors (weight × model)"), t("вывод", "verdict")], rows)
    R.note(t("Родитель и его прямой потомок «объясняют» друг друга одинаково хорошо — кто от кого, "
             "покажут вкладки «Родитель» и «Семья».",
             "A parent and its direct child 'explain' each other equally well — the 'Parent' and 'Family' tabs "
             "show which came first."))
    R.data["decompose"] = {names[i]: {"r2": r2, "r2_per_block": bdecs[i][0],
                                      "weights": {names[j]: x for j, x in keep},
                                      "per_block": {str(b): {names[j]: x for j, x in d.items()}
                                                    for b, d in bdecs[i][1].items()}}
                           for i, (keep, r2) in decs.items()}

    shown = 0
    if nb:
        for i in range(n):
            keep, r2 = decs[i]
            br2, per_block = bdecs[i]
            use_blocks = br2 is not None and br2 - r2 > 0.15
            if shown >= 8 or (not use_blocks and (r2 < 0.5 or not keep)):
                continue
            if use_blocks:
                tot = {}
                for d in per_block.values():
                    for j, x in d.items():
                        tot[j] = tot.get(j, 0) + x
                idx = [j for j, _ in sorted(tot.items(), key=lambda z: -z[1])][:4]
                series = [(names[j], [per_block[b].get(j, 0.0) if b in per_block else None for b in range(nb)])
                          for j in idx]
                ttl = f"{names[i]}  (R² {t('по блокам', 'per block')} {br2:.2f})"
            else:
                idx = [j for j, _ in keep][:4]
                series = []
                for j in idx:
                    vals = []
                    for b in range(nb):
                        g = GB[b]
                        if g[i, i] <= 0:
                            vals.append(None)
                            continue
                        wb = nnls_gram(g[np.ix_(idx, idx)], g[idx, i])
                        vals.append(float(wb[idx.index(j)]))
                    series.append((names[j], vals))
                ttl = f"{names[i]}  (R² {r2:.2f})"
            if not series:
                continue
            if shown == 0:
                R.h3(t("Веса предков по блокам (видно поблочный мердж, MBW)",
                       "Ancestor weights per block (shows block-weighted merges, MBW)"))
            R.lines(block_labels(nb), series, ttl, fmt_s="{:.2f}", x_label=t("блок", "block"), y_min=0)
            shown += 1

    if nb and genes:
        R.h2(t("5. Где живут главные гены", "5. Where the main genes live"))
        tn = type_names()
        for c in genes[:3]:
            v = V[:, c]
            e = np.array([max(v @ GB[b] @ v, 0) for b in range(nb)])
            if e.sum() <= 0:
                continue
            e = e / e.sum()
            R.bars(block_labels(nb), list(100 * e),
                   t(f"Ген {c + 1}: доля по блокам, %", f"Gene {c + 1}: share per block, %"), fmt_s="{:.1f}",
                   highlight=inserted_for(nb), hl_label=hl_legend())
            et = {tp: max(v @ GT[tp] @ v, 0) for tp in TYPES}
            s = sum(et.values()) or 1
            R.note(t("по типам слоёв: ", "by layer type: ") +
                   ", ".join(f"{tn[tp]} {100 * x / s:.0f}%" for tp, x in sorted(et.items(), key=lambda z: -z[1]) if x > 0))

    R.h2(t("6. Размер изменений по типам слоёв", "6. Size of changes by layer type"))
    tn = type_names()
    tps = [tp for tp in TYPES if BS["type"][tp] > 0]
    R.heatmap(names, [tn[tp] for tp in tps],
              [[math.sqrt(max(GT[tp][i, i], 0) / BS["type"][tp]) for tp in tps] for i in range(n)],
              "|delta| / |base|", fmt_s="{:.4f}")
    if nb:
        R.h2(t("7. Размер изменений по блокам", "7. Size of changes per block"))
        R.heatmap(names, block_labels(nb),
                  [[math.sqrt(max(GB[b, i, i], 0) / BS["blk"][b]) if BS["blk"][b] > 0 else None for b in range(nb)]
                   for i in range(n)], "|delta| / |base|", fmt_s="{:.4f}", show_values=False)
    pairs = [(int(SAME[i, j]), names[i], names[j]) for i in range(n) for j in range(i + 1, n) if SAME[i, j] > 0]
    if pairs:
        R.h2(t("8. Одинаковые тензоры у пар моделей", "8. Identical tensors in model pairs"))
        R.note(t("Много совпадений = одна модель сделана из другой без изменения этих слоёв.",
                 "Many matches = one model was made from the other without touching these layers."))
        R.table([t("совпадает тензоров", "identical tensors"), t("модель", "model"), t("модель", "model")],
                sorted(pairs, reverse=True)[:20])
    rec = [(names[i], ", ".join(k for k in ("anima_merger", "sd_merge_recipe", "sd_merge_models", "recipe", "workflow")
                                if k in models[i].meta)) for i in range(n)]
    rec = [r for r in rec if r[1]]
    if rec:
        R.h2(t("9. Рецепты в метаданных", "9. Recipes in metadata"))
        R.note(t("Эти модели хранят рецепт в заголовке файла — смотри вкладку «Рецепт / метаданные».",
                 "These models store a recipe in the file header — see the 'Recipe / metadata' tab."))
        R.table([t("модель", "model"), t("поля", "fields")], rec)
    R.save("genome")


# ------------------------------------------------------------------- blocks --

def cmd_blocks(args):
    dev = pick_device(args.device)
    ex = getattr(args, "exact", False)
    ref = open_src(args.ref, ex)
    paths = [p for p in resolve_models(args.models) if os.path.abspath(p) != os.path.abspath(args.ref)]
    models = load_models(paths, exact=ex)
    models = [m for m in models if m.nblocks == ref.nblocks or (log(t(
        f"  пропуск {short(m.path)}: {m.nblocks} блоков, у эталона {ref.nblocks}",
        f"  skipped {short(m.path)}: {m.nblocks} blocks, the reference has {ref.nblocks}")) and False)]
    if not models:
        raise SystemExit(t("Нет моделей той же архитектуры, что эталон.", "No models of the reference's architecture."))
    s2b = open_src(args.base2b, ex) if args.base2b else None
    if s2b is not None and not (ref.nblocks == 40 and s2b.nblocks == 28):
        log(t("  --base2b нужна только для моделей 2.9B (40 блоков) и должна быть 2B (28) — игнорирую.",
              "  --base2b is only for 2.9B models (40 blocks) and must be 2B (28) — ignored."))
        s2b = None
    n, nb = len(models), max(ref.nblocks, 1)
    names = unique_labels([m.path for m in models], 26)
    d2b = torch.zeros(n, nb, dtype=torch.float64, device=dev)
    r2b = torch.zeros(nb, dtype=torch.float64, device=dev)
    d2t = torch.zeros(n, len(TYPES), dtype=torch.float64, device=dev)
    r2t = torch.zeros(len(TYPES), dtype=torch.float64, device=dev)
    grp_d = {}
    grp_r = {}
    v2b = torch.zeros(n, 2, dtype=torch.float64, device=dev)
    keys = [k for k in ref.map if ref.dtype(k) in FLOAT]
    prog = Progress(len(keys), t("Анализ блоков", "Block analysis"))
    for k in keys:
        prog.step()
        r = ref.get(k, dev)
        shp = ref.shape(k)
        bi, tp = key_type(k)
        ti = TYPES.index(tp)
        rr = (r * r).sum().double()
        r2t[ti] += rr
        if bi is not None:
            r2b[bi] += rr
        g = k.split(".")[0] if bi is None else None
        if g:
            grp_r[g] = grp_r.get(g, 0.0) + float(rr)
        sk = None
        if s2b is not None and bi is not None and bi in OFFICIAL_SMALL_TO_LARGE:
            k2 = f"blocks.{OFFICIAL_SMALL_TO_LARGE.index(bi)}.{BLK.match(k).group(2)}"
            if s2b.has(k2) and s2b.shape(k2) == shp:
                sk = s2b.get(k2, dev)
        for i, m in enumerate(models):
            if not m.has(k) or m.shape(k) != shp:
                continue
            x = m.get(k, dev)
            d = x - r
            s = (d * d).sum().double()
            d2t[i, ti] += s
            if bi is not None:
                d2b[i, bi] += s
            if g:
                grp_d.setdefault(g, np.zeros(n))[i] += float(s)
            if sk is not None:
                e = x - sk
                v2b[i, 0] += (e * e).sum().double()
                v2b[i, 1] += (sk * sk).sum().double()
    d2b, r2b, d2t, r2t, v2b = (x.cpu().numpy() for x in (d2b, r2b, d2t, r2t, v2b))
    ins = inserted_for(ref.nblocks)
    R = Report("blocks", t("Где модели отличаются от эталона", "Where models differ from the reference"),
               t(f"Эталон: {short(ref.path, 60)}" + (f", база 2B: {short(s2b.path, 40)}" if s2b else "") + ". Только чтение.",
                 f"Reference: {short(ref.path, 60)}" + (f", 2B base: {short(s2b.path, 40)}" if s2b else "") + ". Read-only."))
    R.data.update({"ref": ref.path, "models": [m.path for m in models], "exact": ex})
    mode_note(R, ex)
    R.h2(t("Сводка", "Summary"))
    hdr = [t("модель", "model"), t("всего |Δ|/|эталон|", "total |Δ|/|ref|")]
    if ins:
        hdr += [t("старые блоки", "original blocks"), t("вставленные", "inserted"), t("доля во вставленных", "share in inserted")]
    if s2b is not None:
        hdr.append(t("старые блоки vs 2B", "original blocks vs 2B"))
    rows = []
    for i in range(n):
        tot = d2b[i].sum()
        row = [names[i], math.sqrt(d2t[i].sum() / max(r2t.sum(), 1e-30))]
        if ins:
            old = [b for b in range(nb) if b not in ins]
            row += [math.sqrt(d2b[i, old].sum() / max(r2b[old].sum(), 1e-30)),
                    math.sqrt(d2b[i, list(ins)].sum() / max(r2b[list(ins)].sum(), 1e-30)),
                    f"{100 * d2b[i, list(ins)].sum() / max(tot, 1e-30):.0f}%"]
        if s2b is not None:
            row.append(math.sqrt(v2b[i, 0] / v2b[i, 1]) if v2b[i, 1] > 0 else None)
        rows.append(row)
    R.table(hdr, rows, fmts={1: "{:.4f}", 2: "{:.4f}", 3: "{:.4f}", 5: "{:.4f}"})
    if ins:
        R.note(t("12 вставленных блоков 2.9B: 2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36 (отмечены *). "
                 "Если у модели почти вся разница во вставленных блоках — у неё своё «расширение».",
                 "The 12 inserted 2.9B blocks: 2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36 (marked *). "
                 "If almost all of a model's difference is in them, it has its own 'expansion'."))
    if s2b is not None:
        R.note(t("«старые блоки vs 2B» = насколько старые 28 блоков модели отличаются от 2B base. "
                 "0 — их не трогали после расширения.",
                 "'original blocks vs 2B' = how much the model's 28 original blocks differ from the 2B base. "
                 "0 = untouched after the expansion."))
    tn = type_names()
    for i in range(n):
        R.h2(names[i])
        R.bars(block_labels(nb), [math.sqrt(d2b[i, b] / r2b[b]) if r2b[b] > 0 else 0 for b in range(nb)],
               t("Отличие по блокам, |Δ| / |эталон|", "Difference per block, |Δ| / |ref|"), fmt_s="{:.4f}",
               highlight=ins, hl_label=hl_legend())
        tot = max(d2t[i].sum(), 1e-30)
        R.table([t("тип слоёв", "layer type"), t("доля изменений, %", "share of changes, %"), "|Δ| / |ref|"],
                [[tn[tp], 100 * d2t[i, j] / tot, math.sqrt(d2t[i, j] / r2t[j]) if r2t[j] > 0 else None]
                 for j, tp in enumerate(TYPES) if r2t[j] > 0], fmts={1: "{:.1f}", 2: "{:.4f}"})
        if grp_d:
            R.table([t("часть вне блоков", "non-block part"), "|Δ| / |ref|"],
                    [[g, math.sqrt(grp_d[g][i] / grp_r[g]) if grp_r.get(g) else None] for g in sorted(grp_d)],
                    fmts={1: "{:.4f}"})
    R.h2(t("Устройство файлов", "File structure"))
    strip = lambda k: re.sub(r"^blocks\.\d+\.", "blocks.N.", k)
    kr = {strip(k) for k in ref.map}
    rows = []
    for i, m in enumerate(models):
        km = {strip(k) for k in m.map}
        bad = {strip(k) for k in ref.map if m.has(k) and m.shape(k) != ref.shape(k)}
        dts = {}
        for k in list(m.map)[:4000]:
            dt = m.dtype(k)
            dts[dt] = dts.get(dt, 0) + 1
        rows.append([names[i], len(km - kr), len(kr - km), len(bad), ", ".join(f"{a}:{b}" for a, b in dts.items()),
                     " ".join(sorted(m.prefixes)) or "—"])
    R.table([t("модель", "model"), t("лишних ключей", "extra keys"), t("нет ключей", "missing keys"),
             t("другая форма", "other shape"), t("типы", "dtypes"), t("префикс", "prefix")], rows)
    R.save("blocks")


# --------------------------------------------------------------------- meta --

TRAIN_KEYS = [
    ("ss_sd_model_name", "base model", "базовая модель"), ("ss_network_module", "network", "сеть"),
    ("ss_network_dim", "dim", "dim"), ("ss_network_alpha", "alpha", "alpha"),
    ("ss_network_args", "network args", "network args"), ("ss_learning_rate", "LR", "LR"),
    ("ss_unet_lr", "UNet/DiT LR", "LR DiT"), ("ss_text_encoder_lr", "TE LR", "LR TE"),
    ("ss_optimizer", "optimizer", "оптимизатор"), ("ss_lr_scheduler", "scheduler", "планировщик"),
    ("ss_resolution", "resolution", "разрешение"), ("ss_num_train_images", "train images (× repeats)", "картинок (× повторы)"),
    ("ss_num_epochs", "epochs", "эпох"), ("ss_epoch", "epoch of this file", "эпоха файла"),
    ("ss_max_train_steps", "max steps", "всего шагов"), ("ss_steps", "steps of this file", "шагов у файла"),
    ("ss_batch_size_per_device", "batch", "batch"), ("ss_gradient_accumulation_steps", "grad accumulation", "накопление градиента"),
    ("ss_caption_dropout_rate", "caption dropout", "caption dropout"), ("ss_noise_offset", "noise offset", "noise offset"),
    ("ss_min_snr_gamma", "min SNR gamma", "min SNR gamma"), ("ss_seed", "seed", "сид"),
    ("ss_training_started_at", "started", "начало"), ("ss_training_finished_at", "finished", "конец"),
]


def pretty_json(s, limit=8000):
    try:
        out = json.dumps(json.loads(s), ensure_ascii=False, indent=2)
    except Exception:
        out = str(s)
    return out if len(out) <= limit else out[:limit] + "\n…"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_meta(args):
    files = resolve_models(args.files)
    if not files:
        raise SystemExit(t("Нет файлов.", "No files."))
    R = Report("meta", t("Рецепты и метаданные", "Recipes and metadata"),
               t("Читаются только заголовки файлов — мгновенно и без изменений.",
                 "Only file headers are read — instant and read-only."))
    rows, cards = [], []
    prog = Progress(len(files), t("Заголовки", "Headers"))
    for p in files:
        prog.step()
        try:
            hdr, meta = read_header(p)
        except Exception as e:
            rows.append([short(p, 40), "?", "", "", "", str(e)[:40]])
            continue
        info = file_kind(p)
        h = sha256(p)[:12] if args.sha256 else ""
        has = [x for x in ("anima_merger", "sd_merge_recipe", "sd_merge_models") if x in meta]
        if any(k.startswith("ss_") for k in meta):
            has.append(t("параметры обучения", "training parameters"))
        rows.append([short(p, 40), info["kind"], info["blocks"], f"{info['size'] / 2**30:.2f}", h,
                     ", ".join(has) or "—"])
        cards.append((p, meta, info))
    R.table([t("файл", "file"), t("тип", "kind"), t("блоков", "blocks"), t("ГБ", "GB"), "sha256", t("рецепт / обучение", "recipe / training")],
            rows)
    for p, meta, info in cards:
        if not meta:
            continue
        R.h2(short(p, 80))
        title = meta.get("modelspec.title")
        if title or meta.get("modelspec.author"):
            R.note(" · ".join(x for x in (title, meta.get("modelspec.author"), meta.get("modelspec.date")) if x))
        if "anima_merger" in meta:
            R.h3(t("Как сделана (AnimaMerger)", "How it was made (AnimaMerger)"))
            R.pre(pretty_json(meta["anima_merger"]))
        for k in ("sd_merge_recipe", "sd_merge_models", "recipe"):
            if k in meta:
                R.h3(k)
                R.pre(pretty_json(meta[k], 6000))
        tr = [(lab_ru if t("ru", "en") == "ru" else lab_en, meta[k]) for k, lab_en, lab_ru in TRAIN_KEYS if k in meta]
        if tr:
            R.h3(t("Параметры обучения", "Training parameters"))
            R.table([t("параметр", "parameter"), t("значение", "value")], [[a, str(b)[:120]] for a, b in tr])
        if meta.get("modelspec.description"):
            R.details(t("Описание", "Description"), f"<pre>{meta['modelspec.description'][:3000]}</pre>")
        other = sorted(k for k in meta if not k.startswith(("ss_", "modelspec.")) and k not in
                       ("anima_merger", "sd_merge_recipe", "sd_merge_models", "recipe", "workflow", "prompt"))
        if other:
            R.note(t("Прочие поля: ", "Other fields: ") + ", ".join(other[:30]))
        R.data.setdefault("files", {})[p] = {k: (v[:4000] if isinstance(v, str) else v) for k, v in meta.items()
                                             if k != "modelspec.thumbnail"}
    R.save("meta")


# -------------------------------------------------------------- LoRA analyzer --

JUNK_TAGS = {"signature", "watermark", "text", "artist name", "patreon username", "patreon logo", "twitter username",
             "web address", "dated", "copyright name", "character name", "english text", "speech bubble",
             "jpeg artifacts", "lowres", "blurry", "border", "letterboxed", "logo", "sample watermark", "translated",
             "commentary", "commentary request", "scan", "scan artifacts", "artist logo", "pixiv username",
             "censored", "bar censor", "mosaic censoring", "japanese text", "chinese text", "korean text",
             "subtitled", "url", "username", "copyright notice", "page number", "multiple views", "comic",
             "4koma", "white border", "black border", "frame", "photo (medium)", "3d", "ai-generated", "ai generated"}

STEP_PATTERNS = [
    (re.compile(r"[-_]step[-_]?0*(\d+)$", re.I), "step"),
    (re.compile(r"[-_](?:epoch|ep|e)[-_]?0*(\d+)$", re.I), "epoch"),
    (re.compile(r"[-_]0*(\d{6})$"), "epoch"),
    (re.compile(r"[-_]0*(\d+)$"), "num"),
]


def detect_step(path, meta):
    stem = Path(path).stem
    for rx, kind in STEP_PATTERNS:
        m = rx.search(stem)
        if m:
            return int(m.group(1)), kind
    for k, kind in (("ss_steps", "step"), ("ss_epoch", "epoch")):
        if meta.get(k):
            try:
                return int(float(meta[k])), kind + "-meta"
            except ValueError:
                pass
    return None, "final"


def lora_eff_rank(f, g, scale, dev):
    """Effective rank of up@down via small factors (cheap)."""
    up = f.get_tensor(g["up"]).to(dev).float()
    down = f.get_tensor(g["down"]).to(dev).float()
    up = up.reshape(up.shape[0], -1)
    down = down.reshape(down.shape[0], -1)
    try:
        _, R1 = torch.linalg.qr(up)
        _, R2 = torch.linalg.qr(down.T)
        s = torch.linalg.svdvals(R1 @ R2.T) * scale
    except Exception:
        return None, None
    p = s ** 2
    tot = p.sum()
    if tot <= 0:
        return None, None
    p = p / tot
    ent = -(p * torch.log(p + 1e-30)).sum()
    return float(torch.exp(ent)), int(down.shape[0])


def cmd_lora_info(args):
    from lora_tools import group_lora, lora_delta, TE_PREFIXES, TargetIndex
    from anima_merger import Model
    dev = pick_device(args.device)
    files = []
    for p in args.loras:
        if os.path.isdir(p):
            files += [str(x) for x in sorted(Path(p).glob("*.safetensors")) if file_kind(str(x))["kind"] == "lora"]
        else:
            files.append(p)
    if not files:
        raise SystemExit(t("Нет файлов LoRA.", "No LoRA files."))
    L = []
    for p in files:
        f, groups, unknown, md = group_lora(p)
        step, kind = detect_step(p, md)
        L.append({"path": p, "f": f, "groups": groups, "meta": md, "step": step, "kind": kind})
    manual = [x.strip() for x in (args.steps or "").replace(";", ",").split(",") if x.strip()]
    if manual:
        L.sort(key=lambda x: Path(x["path"]).name.lower())
        if len(manual) != len(L):
            raise SystemExit(t(f"Указано {len(manual)} номеров для {len(L)} файлов.",
                               f"{len(manual)} numbers given for {len(L)} files."))
        for x, s in zip(L, manual):
            x["step"], x["kind"] = int(float(s)), "manual"
    else:
        nums = [x["step"] for x in L if x["step"] is not None]
        top = max(nums) if nums else 0
        for x in L:
            if x["step"] is None:
                ms = x["meta"].get("ss_max_train_steps") or x["meta"].get("ss_num_epochs")
                try:
                    guess = int(float(ms)) if ms else None
                except ValueError:
                    guess = None
                x["step"] = guess if (guess and guess > top) else top + 1
                x["final"] = True
        L.sort(key=lambda x: (x["step"], x.get("final", False)))
    n = len(L)
    unit = t("шаг", "step")
    kinds = {x["kind"] for x in L}
    if kinds & {"epoch", "epoch-meta"} and not kinds & {"step", "step-meta"}:
        unit = t("эпоха", "epoch")
    xlabels = [(t("финал", "final") if x.get("final") else str(x["step"])) for x in L]
    log(t("Файлы по порядку: ", "Files in order: ") + ", ".join(f"{short(x['path'], 30)} [{lab}]" for x, lab in zip(L, xlabels)))

    model = Model(args.model, "model") if args.model else None
    tidx = TargetIndex(model) if model else None
    names = sorted(set().union(*[set(x["groups"]) for x in L]))
    names = [nm for nm in names if not any(nm.lower().startswith(p) for p in TE_PREFIXES)]
    nb = 0
    from lora_tools import BLOCK_IN_NAME
    for nm in names:
        mm = BLOCK_IN_NAME.search(nm)
        if mm and "llm_adapter" not in nm:
            nb = max(nb, int(mm.group(1)) + 1)
    G = np.zeros((n, n))
    EB = np.zeros((n, max(nb, 1)))
    ET = np.zeros((n, len(TYPES)))
    rel = [[] for _ in range(n)]
    er_num = np.zeros(n)
    er_den = np.zeros(n)
    nom_rank = [None] * n
    missing_model = 0
    prog = Progress(len(names), t("Анализ LoRA", "LoRA analysis"))
    for nm in names:
        prog.step()
        mm = BLOCK_IN_NAME.search(nm)
        bi = int(mm.group(1)) if (mm and "llm_adapter" not in nm) else None
        rest = nm[mm.end(1) + 1:] if mm else nm
        tp = "adapter" if "llm_adapter" in nm else (ltype(rest) if bi is not None else "nonblock")
        idx, ds = [], []
        for i, x in enumerate(L):
            g = x["groups"].get(nm)
            if not g:
                continue
            try:
                base = None
                if "dora" in g and tidx is not None:  # DoRA: exact only with the model
                    mk = tidx.resolve(nm)
                    base = model.get(mk) if mk else None
                d, kd = lora_delta(x["f"], g, None, dev, base=base)
            except Exception:
                continue
            if d is None:
                continue
            d = d.reshape(-1)
            if ds and d.numel() != ds[0].numel():
                continue
            idx.append(i)
            ds.append(d)
            e2 = float((d * d).sum())
            if kd == "lora":
                alpha = float(x["f"].get_tensor(g["alpha"]).float()) if "alpha" in g else None
                r = x["f"].get_slice(g["down"]).get_shape()[0]
                er, rr = lora_eff_rank(x["f"], g, (alpha / r) if alpha else 1.0, dev)
                if er is not None:
                    er_num[i] += er * e2
                    er_den[i] += e2
                    nom_rank[i] = rr
        if not ds:
            continue
        Y = torch.stack(ds)
        g_ = gram(Y).cpu().numpy()
        G[np.ix_(idx, idx)] += g_
        for a, i in enumerate(idx):
            if bi is not None:
                EB[i, bi] += g_[a, a]
            ET[i, TYPES.index(tp)] += g_[a, a]
        if tidx is not None:
            k = tidx.resolve(nm)
            if k is None:
                missing_model += 1
            else:
                wn = float(model.get(k).float().norm())
                if wn > 0:
                    for a, i in enumerate(idx):
                        rel[i].append(math.sqrt(g_[a, a]) / wn)
        del Y, ds

    norm = np.sqrt(np.maximum(np.diag(G), 0))
    F = n - 1
    cosF = [G[i, F] / (norm[i] * norm[F]) if norm[i] > 0 and norm[F] > 0 else None for i in range(n)]
    cosP = [None] + [G[i, i - 1] / (norm[i] * norm[i - 1]) if norm[i] > 0 and norm[i - 1] > 0 else None for i in range(1, n)]
    chg = [None] + [math.sqrt(max(G[i, i] + G[i - 1, i - 1] - 2 * G[i, i - 1], 0)) / norm[i] if norm[i] > 0 else None
                    for i in range(1, n)]
    eff = [er_num[i] / er_den[i] if er_den[i] > 0 else None for i in range(n)]

    title = short(L[-1]["path"], 60) if n == 1 else t(f"{n} чекпоинтов: ", f"{n} checkpoints: ") + short(L[-1]["path"], 50)
    R = Report("lora", t("Анализ LoRA — ", "LoRA analysis — ") + title,
               t("Анализ по весам: что и насколько сильно меняет LoRA. Выводы про недо/переобучение — эвристики, "
                 "окончательно решают картинки на одном сиде.",
                 "Weight-based analysis: what the LoRA changes and how strongly. Under/over-training verdicts are "
                 "heuristics — images on a fixed seed have the final word."))
    R.data.update({"files": [x["path"] for x in L], "steps": xlabels, "norm": norm.tolist(), "cos_final": cosF,
                   "cos_prev": cosP, "change": chg, "eff_rank": eff})
    R.h2(t("Вывод", "Verdict"))
    settle = None
    if n >= 3:
        for i in range(n - 1):
            if cosF[i] is not None and cosF[i] >= 0.95:
                settle = i
                break
        growth = norm[F] / norm[settle] - 1 if settle is not None and norm[settle] > 0 else None
        if cosF[n - 2] is not None and cosF[n - 2] < 0.9:
            R.badge("warn", t(f"К концу обучения направление ещё меняется (похожесть предпоследнего чекпоинта на финал "
                              f"{cosF[n - 2]:.2f}). LoRA, возможно, недоучена: больше шагов могли бы помочь.",
                              f"The direction still changes at the end (second-to-last vs final similarity "
                              f"{cosF[n - 2]:.2f}). The LoRA may be undertrained: more steps could help."))
        elif settle is not None:
            R.badge("info", t(f"Направление сложилось к {unit} {xlabels[settle]}: дальше LoRA учит в основном то же самое.",
                              f"The direction settled by {unit} {xlabels[settle]}: after that the LoRA mostly learns the same thing."))
            if growth is not None and growth > 0.3:
                R.badge("warn", t(f"После этого сила выросла ещё на {100 * growth:.0f}% без нового содержания — риск пережога. "
                                  f"Попробуй чекпоинты около {xlabels[settle]}–{xlabels[min(n - 1, settle + 2)]} "
                                  f"или финал с меньшим весом.",
                                  f"After that the strength grew another {100 * growth:.0f}% with no new content — risk of "
                                  f"over-baking. Try checkpoints around {xlabels[settle]}–{xlabels[min(n - 1, settle + 2)]} "
                                  f"or the final one at a lower weight."))
            else:
                R.badge("good", t("После этого сила растёт умеренно: финальный чекпоинт выглядит безопасным выбором.",
                                  "After that the strength grows moderately: the final checkpoint looks like a safe pick."))
    meds = [float(np.median(r)) if r else None for r in rel]
    if model is not None and meds[F] is not None:
        p90 = float(np.percentile(rel[F], 90))
        R.data.update({"rel_median": meds, "rel_p90_final": p90})
        lvl = "good" if meds[F] < 0.02 else ("warn" if meds[F] < 0.06 else "bad")
        R.badge(lvl, t(f"Сила относительно модели (вес 1): медиана изменения слоя {100 * meds[F]:.1f}%, "
                       f"у 10% самых сильных слоёв > {100 * p90:.1f}%. "
                       + ("Мягкая LoRA." if lvl == "good" else "Заметная, обычно ок на 0.7–1." if lvl == "warn"
                          else "Очень сильная: начинай с веса 0.5–0.7."),
                       f"Strength relative to the model (weight 1): median layer change {100 * meds[F]:.1f}%, "
                       f"top 10% of layers > {100 * p90:.1f}%. "
                       + ("A gentle LoRA." if lvl == "good" else "Noticeable, usually fine at 0.7–1." if lvl == "warn"
                          else "Very strong: start at weight 0.5–0.7.")))
        if missing_model:
            R.note(t(f"Модулей LoRA, не найденных в модели: {missing_model} (LoRA от другой архитектуры?)",
                     f"LoRA modules not found in the model: {missing_model} (another architecture?)"))
    if nom_rank[F] and eff[F] and eff[F] < 0.5 * nom_rank[F]:
        R.note(t(f"Эффективный ранг: {eff[F]:.1f} из {nom_rank[F]}. Сильно меньше номинального — LoRA сжалась в "
                 f"несколько направлений (ранг можно уменьшить; иногда признак узкого датасета).",
                 f"Effective rank: {eff[F]:.1f} of {nom_rank[F]}. Much lower than nominal — the LoRA squeezed into a few "
                 f"directions (the rank could be lower; sometimes a sign of a narrow dataset)."))
    if n >= 2:
        R.h2(t("Обучение по чекпоинтам", "Training over checkpoints"))
        marks = []
        xs = [x["step"] for x in L]
        if settle is not None:
            marks.append((xs[settle], t("выучилась", "learned")))
            gi = None
            for i in range(settle + 1, n):
                if norm[settle] > 0 and norm[i] / norm[settle] - 1 > 0.3:
                    gi = i
                    break
            if gi is not None:
                marks.append((xs[gi], t("дальше в основном сила", "mostly strength after")))
        R.lines(xs, [(t("сила LoRA |ΔW|", "LoRA strength |ΔW|"), list(norm))],
                t("Сила LoRA по чекпоинтам", "LoRA strength per checkpoint"), x_label=unit, markers=marks, y_min=0)
        R.lines(xs, [(t("похожесть на финал", "similarity to final"), cosF),
                     (t("похожесть на предыдущий", "similarity to previous"), cosP)],
                t("Куда смотрит LoRA (1 = то же направление)", "Where the LoRA points (1 = same direction)"),
                fmt_s="{:.2f}", x_label=unit, y_min=0, y_max=1, markers=marks)
        R.lines(xs, [(t("изменение за шаг", "change per step"), chg)],
                t("Насколько чекпоинт изменился с предыдущего (доля)", "How much each checkpoint changed since the previous (share)"),
                fmt_s="{:.2f}", x_label=unit, y_min=0)
        R.note(t("Как читать: пока «похожесть на финал» растёт — LoRA ещё учится новому. Когда она около 1, а сила "
                 "продолжает расти — модель уже не учит новое, а «пережимает» выученное: тут обычно и начинается пережог.",
                 "How to read: while 'similarity to final' rises, the LoRA still learns new things. When it's near 1 but the "
                 "strength keeps growing, the model isn't learning anything new, it's pushing harder: that's where "
                 "over-baking usually starts."))
        R.table([unit, t("файл", "file"), t("сила", "strength"), t("к финалу", "to final"), t("к предыдущему", "to previous"),
                 t("эфф. ранг", "eff. rank")] + ([t("изм. слоя (медиана)", "layer change (median)")] if model else []),
                [[xlabels[i], short(L[i]["path"], 36), float(norm[i]), cosF[i], cosP[i], eff[i]]
                 + ([meds[i]] if model else []) for i in range(n)],
                fmts={2: "{:.3f}", 3: "{:.3f}", 4: "{:.3f}", 5: "{:.1f}", 6: "{:.4f}"})
    if nb:
        R.h2(t("Где работает LoRA", "Where the LoRA works"))
        tot = max(EB[F].sum(), 1e-30)
        R.bars(block_labels(nb), list(100 * EB[F] / tot),
               t("Доля силы по блокам (финал), %", "Share of strength per block (final), %"), fmt_s="{:.1f}",
               highlight=inserted_for(nb), hl_label=hl_legend())
        if n >= 2:
            R.heatmap(xlabels, block_labels(nb),
                      [[100 * EB[i, b] / max(EB[i].sum(), 1e-30) for b in range(nb)] for i in range(n)],
                      t("Доля по блокам у каждого чекпоинта, %", "Share per block for each checkpoint, %"), fmt_s="{:.0f}",
                      show_values=nb <= 28)
        tn = type_names()
        tt = max(ET[F].sum(), 1e-30)
        R.table([t("тип слоёв", "layer type"), "%"],
                [[tn[tp], 100 * ET[F, j] / tt] for j, tp in enumerate(TYPES) if ET[F, j] > 0], fmts={1: "{:.1f}"})
        R.note(t("cross_attn — как LoRA «понимает» промпт и триггер; mlp и self_attn — как рисует; adaLN — общий "
                 "контраст/свет. Блоки: ранние — композиция и формы, поздние — детали и фактура.",
                 "cross_attn — how the LoRA 'hears' the prompt and trigger; mlp and self_attn — how it draws; adaLN — global "
                 "contrast/light. Blocks: early — composition and shapes, late — details and texture."))
    meta = next((x["meta"] for x in reversed(L) if x["meta"].get("ss_tag_frequency")), L[-1]["meta"])
    tr = [(lab_ru if t("ru", "en") == "ru" else lab_en, meta[k]) for k, lab_en, lab_ru in TRAIN_KEYS if k in meta]
    if tr:
        R.h2(t("Параметры обучения (из метаданных)", "Training parameters (from metadata)"))
        R.table([t("параметр", "parameter"), t("значение", "value")], [[a, str(b)[:120]] for a, b in tr])
    if meta.get("ss_tag_frequency"):
        try:
            tf = json.loads(meta["ss_tag_frequency"])
        except Exception:
            tf = {}
        counts = {}
        for d in tf.values():
            for tag, c in (d or {}).items():
                tag = tag.strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + int(c)
        n_img = 0
        try:
            dirs = json.loads(meta.get("ss_dataset_dirs", "{}"))
            n_img = sum(int(v.get("img_count", 0)) for v in dirs.values())
        except Exception:
            pass
        n_img = n_img or (max(counts.values()) if counts else 0)
        if counts and n_img:
            R.h2(t("Датасет по тегам", "Dataset by tags"))
            R.note(t(f"Картинок: {n_img}, разных тегов: {len(counts)}.", f"Images: {n_img}, distinct tags: {len(counts)}."))
            top = sorted(counts.items(), key=lambda x: -x[1])[:25]
            R.bars([a for a, _ in top], [100 * b / n_img for _, b in top],
                   t("Самые частые теги, % картинок", "Most frequent tags, % of images"), fmt_s="{:.0f}", horizontal=True)
            always = [a for a, b in counts.items() if b >= 0.9 * n_img]
            if always:
                R.badge("info", t("Почти на всех картинках: ", "On almost every image: ") + ", ".join(always[:15]) +
                        t(". Эти теги LoRA свяжет со стилем: без них в промпте эффект слабее.",
                          ". The LoRA ties these to the style: without them in the prompt the effect is weaker."))
            junk = sorted((a, b) for a, b in counts.items() if a.lower().replace("_", " ") in JUNK_TAGS)
            if junk:
                R.badge("warn", t("В датасете есть «мусорные» теги: ", "The dataset has 'junk' tags: ") +
                        ", ".join(f"{a} ({b})" for a, b in junk[:15]) +
                        t(". Значит, на картинках есть то, что LoRA тоже выучит (подписи, вотермарки, рамки…). "
                          "Почисти картинки или хотя бы оставь эти теги в подписях, чтобы их можно было убрать негативом.",
                          ". So the images contain things the LoRA will learn too (signatures, watermarks, borders…). "
                          "Clean the images, or at least keep these tags in captions so a negative can remove them."))
            rare = sum(1 for b in counts.values() if b == 1)
            if rare > 0.5 * len(counts) and len(counts) > 40:
                R.note(t(f"Тегов, встречающихся один раз: {rare} — подписи очень разнообразные, это нормально для стиля.",
                         f"Tags used only once: {rare} — very diverse captions, normal for a style LoRA."))
    R.save("lora")


# ---------------------------------------------------------------------- CLI --

def add_parsers(sub):
    dev_help = t("auto / cuda / cpu", "auto / cuda / cpu")
    p = sub.add_parser("parent", help=t("найти родителя модели", "find a model's parent"))
    p.add_argument("--target", required=True)
    p.add_argument("--cands", nargs="+", required=True, help=t("кандидаты: файлы или папки", "candidates: files or folders"))
    p.add_argument("--base", default=None, help=t("официальная база (для масштаба и косинусов)", "official base (for scale and cosines)"))
    p.add_argument("--device", default="auto", help=dev_help)
    p.add_argument("--exact", action="store_true", help=t("точно, без профилей: читать модели целиком (медленно)", "exact, no profiles: read models whole (slow)"))
    p.set_defaults(func=cmd_parent)

    p = sub.add_parser("family", help=t("семейное древо моделей", "family tree of models"))
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ref", required=True, help=t("эталон (обычно официальная база)", "reference (usually the official base)"))
    p.add_argument("--device", default="auto", help=dev_help)
    p.add_argument("--exact", action="store_true", help=t("точно, без профилей: читать модели целиком (медленно)", "exact, no profiles: read models whole (slow)"))
    p.set_defaults(func=cmd_family)

    p = sub.add_parser("genome", help=t("гены семейства и состав мерджей", "family genes and merge composition"))
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--device", default="auto", help=dev_help)
    p.add_argument("--exact", action="store_true", help=t("точно, без профилей: читать модели целиком (медленно)", "exact, no profiles: read models whole (slow)"))
    p.set_defaults(func=cmd_genome)

    p = sub.add_parser("blocks", help=t("отличия от эталона по блокам и слоям", "differences from a reference per block and layer"))
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--base2b", default=None, help=t("для 2.9B: база 2B, чтобы сравнить старые блоки",
                                                    "for 2.9B: a 2B base to compare original blocks"))
    p.add_argument("--device", default="auto", help=dev_help)
    p.add_argument("--exact", action="store_true", help=t("точно, без профилей: читать модели целиком (медленно)", "exact, no profiles: read models whole (slow)"))
    p.set_defaults(func=cmd_blocks)

    p = sub.add_parser("meta", help=t("рецепты и метаданные из заголовков", "recipes and metadata from headers"))
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--sha256", action="store_true", help=t("посчитать SHA256 (медленно)", "compute SHA256 (slow)"))
    p.set_defaults(func=cmd_meta)

    p = sub.add_parser("lora-info", help=t("анализ LoRA (несколько чекпоинтов = график по эпохам)",
                                           "LoRA analysis (several checkpoints = curve over epochs)"))
    p.add_argument("--loras", nargs="+", required=True, help=t("файлы или папка", "files or a folder"))
    p.add_argument("--steps", default="", help=t("номера шагов/эпох вручную, по алфавиту файлов: '500,1000,1500'",
                                                 "manual step/epoch numbers in file-name order: '500,1000,1500'"))
    p.add_argument("--model", default=None, help=t("модель для оценки силы относительно её весов",
                                                   "a model to measure strength against its weights"))
    p.add_argument("--device", default="auto", help=dev_help)
    p.set_defaults(func=cmd_lora_info)

    profiles.add_parsers(sub)
