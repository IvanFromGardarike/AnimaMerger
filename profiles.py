# -*- coding: utf-8 -*-
"""Model profiles: a small cached "snapshot" of a checkpoint for fast analysis.

A profile is built once per file (one full read) and stored in profiles/:
  <id>.json — file fingerprint, structure (keys, shapes, dtypes), exact per-tensor hashes, header metadata;
  <id>.npz  — a fixed sample of SKETCH elements from every float tensor (raw values, original dtype).

The sample positions depend only on the layer name (block number stripped) and the tensor size, so they are
the same in every model. Sums of squares, dot products and distances computed on the samples (scaled by
sqrt(N / SKETCH)) are unbiased estimates of the exact ones; aggregated over a whole model or block the error
is well under 1–2%. Identical tensors are detected exactly, by hash.

The id is a hash of the file size, the header and the first/last 4 MB, so a renamed or moved file keeps its
profile, and a changed file gets a new one automatically.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from amutil import ROOT, PREFIXES, t, log, Progress, short

SKETCH = 4096
VERSION = 1
FLOAT = {"BF16", "F16", "F32", "F64"}
BLK = re.compile(r"^blocks\.(\d+)\.(.+)$")
_TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
_TINT = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32,
         torch.float64: torch.int64}


def profiles_dir():
    d = ROOT / "profiles"
    d.mkdir(exist_ok=True)
    return d


def quick_id(path):
    """Fingerprint of a file without reading it whole: size + header + first/last 4 MB of the data."""
    h = hashlib.blake2b(digest_size=12)
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h.update(f.read(min(n, 64 * 1024 * 1024)))
        chunk = 4 * 1024 * 1024
        h.update(f.read(chunk))
        if size > 8 + n + 2 * chunk:
            # 64 small probes spread over the data: catches block-wise re-merges with the same header
            start, span = 8 + n + chunk, size - (8 + n + 2 * chunk)
            for i in range(64):
                f.seek(start + span * i // 64)
                h.update(f.read(64 * 1024))
            f.seek(size - chunk)
            h.update(f.read(chunk))
    return h.hexdigest()


def _norm_key(k):
    for p in PREFIXES:
        if k.startswith(p):
            return k[len(p):], p
    return k, None


def _sample_idx(nk, numel):
    """Sample positions: the same for the same layer name (block number stripped) and size in any model."""
    name = re.sub(r"^blocks\.\d+\.", "blocks.N.", nk)
    seed = int.from_bytes(hashlib.blake2b(f"{name}|{numel}".encode(), digest_size=8).digest(), "little") >> 1
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, numel, (SKETCH,), generator=g)


def _tensor_hash(x):
    x = x.contiguous().reshape(-1)
    raw = x.view(_TINT.get(x.dtype, torch.uint8)) if x.dtype in _TINT else x
    return hashlib.blake2b(raw.numpy().tobytes(), digest_size=16).hexdigest()


def build(path, pid=None):
    """Read the whole file once and write its profile. Returns the profile id."""
    pid = pid or quick_id(path)
    d = profiles_dir()
    f = safe_open(str(path), framework="pt", device="cpu")
    meta = f.metadata() or {}
    keys = list(f.keys())
    entries, arrays = [], {}
    prefixes = set()
    prog = Progress(len(keys), t("Профиль: ", "Profile: ") + short(str(path), 40))
    for k in keys:
        prog.step()
        sl = f.get_slice(k)
        dt = sl.get_dtype()
        shape = list(sl.get_shape())
        nk, p = _norm_key(k)
        if p:
            prefixes.add(p)
        e = {"k": nk, "orig": k, "shape": shape, "dtype": dt}
        if dt in FLOAT:
            x = f.get_tensor(k)
            numel = x.numel()
            e["hash"] = _tensor_hash(x)
            flat = x.reshape(-1)
            if numel <= SKETCH:
                s = flat.clone()
                e["n"], e["s"] = numel, numel
            else:
                s = flat[_sample_idx(nk, numel)]
                e["n"], e["s"] = numel, SKETCH
            e["i"] = len(arrays)
            arrays[f"t{e['i']}"] = s.contiguous().view(_TINT[s.dtype]).numpy() if s.dtype in _TINT else s.numpy()
            del x, flat, s
        entries.append(e)
    info = {"version": VERSION, "sketch": SKETCH, "id": pid, "path": str(path),
            "size": os.path.getsize(path), "mtime": os.path.getmtime(path), "built": time.time(),
            "prefixes": sorted(prefixes), "meta": meta, "keys": entries}
    tmp_npz, tmp_json = d / f"{pid}.npz.part", d / f"{pid}.json.part"
    with open(tmp_npz, "wb") as fh:
        np.savez(fh, **arrays)
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False)
    os.replace(tmp_npz, d / f"{pid}.npz")
    os.replace(tmp_json, d / f"{pid}.json")
    return pid


def load_info(pid):
    p = profiles_dir() / f"{pid}.json"
    if not p.exists() or not (profiles_dir() / f"{pid}.npz").exists():
        return None
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if info.get("version") != VERSION or info.get("sketch") != SKETCH:
        return None
    return info


def ensure(path, rebuild=False):
    """Profile id for the file, building it if missing or outdated."""
    pid = quick_id(path)
    info = None if rebuild else load_info(pid)
    if info is not None and os.path.abspath(info.get("path", "")) == os.path.abspath(str(path)) \
            and abs(info.get("mtime", 0) - os.path.getmtime(path)) > 2:
        info = None  # same file, rewritten in place: rebuild to be safe
    if info is None:
        log(t(f"  строю профиль: {short(str(path), 60)} (один раз, дальше берётся из кэша)",
              f"  building profile: {short(str(path), 60)} (once, cached afterwards)"))
        build(path, pid)
        info = load_info(pid)
    elif os.path.abspath(info.get("path", "")) != os.path.abspath(str(path)):
        info["path"] = str(path)  # moved/renamed file: same contents, keep the profile
    return pid, info


class Sketch:
    """Same interface as analysis.Src, but values come from the profile sample (scaled)."""
    sketch = True

    def __init__(self, path, rebuild=False):
        self.path = str(path)
        self.pid, info = ensure(path, rebuild)
        self.meta = info.get("meta") or {}
        self.map, self._e = {}, {}
        for e in info["keys"]:
            self.map[e["k"]] = e["orig"]
            self._e[e["k"]] = e
        self.prefixes = set(info.get("prefixes") or [])
        nbs = {int(m.group(1)) for k in self.map for m in [BLK.match(k)] if m}
        self.nblocks = (max(nbs) + 1) if nbs else 0
        self._z = np.load(profiles_dir() / f"{self.pid}.npz")
        self._cache = {}

    def has(self, k):
        return k in self.map

    def shape(self, k):
        return tuple(self._e[k]["shape"])

    def dtype(self, k):
        return self._e[k]["dtype"]

    def fp(self, k):
        return self._e[k].get("hash")

    def get(self, k, dev):
        e = self._e[k]
        a = self._z[f"t{e['i']}"]
        td = _TORCH[e["dtype"]]
        x = torch.from_numpy(a.copy())
        if td in _TINT:
            x = x.view(_TINT[td]).view(td)
        x = x.to(dev).float()
        if e["s"] < e["n"]:
            x = x * float(np.sqrt(e["n"] / e["s"]))
        return x


def cmd_profile(args):
    from analysis import resolve_models
    from amutil import file_kind
    paths = [p for p in resolve_models(args.models) if file_kind(p)["kind"] == "model"]
    if not paths:
        raise SystemExit(t("Нет моделей.", "No models."))
    built = cached = 0
    for p in paths:
        pid = quick_id(p)
        if not args.rebuild and load_info(pid) is not None:
            cached += 1
            log(t(f"  уже есть: {short(p, 60)}", f"  cached: {short(p, 60)}"))
            continue
        build(p, pid)
        built += 1
    log(t(f"Готово. Построено профилей: {built}, уже были: {cached}. Папка: {profiles_dir()}",
          f"Done. Profiles built: {built}, already cached: {cached}. Folder: {profiles_dir()}"))


def add_parsers(sub):
    p = sub.add_parser("profile", help=t("построить профили моделей для быстрого анализа",
                                         "build model profiles for fast analysis"))
    p.add_argument("--models", nargs="+", required=True, help=t("файлы или папки", "files or folders"))
    p.add_argument("--rebuild", action="store_true", help=t("пересчитать, даже если профиль есть",
                                                            "rebuild even if a profile exists"))
    p.set_defaults(func=cmd_profile)
