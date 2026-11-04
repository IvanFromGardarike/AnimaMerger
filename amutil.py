# -*- coding: utf-8 -*-
"""Shared helpers: language, logging, progress, device choice, model discovery."""
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = ROOT / "settings.json"


# ------------------------------------------------------------------ language --

def _detect_lang():
    v = os.environ.get("ANIMA_LANG", "").strip().lower()
    if v in ("ru", "en"):
        return v
    try:
        v = str(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("lang", "")).lower()
        if v in ("ru", "en"):
            return v
    except Exception:
        pass
    return "en"


LANG = _detect_lang()


def t(ru, en):
    """Pick the string for the current language."""
    return ru if LANG == "ru" else en


# ------------------------------------------------------------------- logging --

def log(*a):
    print(*a, flush=True)


PIPE = os.environ.get("ANIMA_PROGRESS") == "1"   # set by the UI: machine-readable progress lines


def _fmt_time(s):
    s = int(max(0, s))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Progress:
    """Progress for long loops. In the UI it drives the progress bar; in a console it prints a
    line every `every` (10%) with elapsed time and ETA."""

    def __init__(self, total, label, every=0.1):
        self.total = max(1, int(total))
        self.label = label
        self.every = every
        self.done = 0
        self.t0 = time.time()
        self.next_log = every
        self.last_pipe = -1.0
        self._emit(0.0)

    def _emit(self, frac):
        if PIPE:
            print(f"@@PROGRESS {frac:.4f} {self.label}", flush=True)

    def step(self, n=1):
        self.done = min(self.total, self.done + n)
        frac = self.done / self.total
        if PIPE and (frac - self.last_pipe >= 0.004 or self.done == self.total):
            self.last_pipe = frac
            self._emit(frac)
        if frac + 1e-9 >= self.next_log:
            el = time.time() - self.t0
            if el < 2 and frac < 1:      # don't spam the log for fast operations
                return
            if el < 2:
                while self.next_log <= frac + 1e-9:
                    self.next_log += self.every
                return
            eta = el / frac * (1 - frac) if frac > 0 else 0
            log(f"  {self.label}: {100 * frac:3.0f}%   " +
                t(f"прошло {_fmt_time(el)}, осталось ~{_fmt_time(eta)}",
                  f"elapsed {_fmt_time(el)}, ~{_fmt_time(eta)} left"))
            while self.next_log <= frac + 1e-9:
                self.next_log += self.every

    def finish(self):
        if self.done < self.total:
            self.step(self.total - self.done)


def emit_report(path):
    """Tell the UI that a report file is ready."""
    if PIPE:
        print(f"@@REPORT {path}", flush=True)


# -------------------------------------------------------------------- device --

def pick_device(arg="auto"):
    import torch
    arg = (arg or "auto").lower()
    if arg == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif arg.startswith("cuda") and not torch.cuda.is_available():
        log(t("ВНИМАНИЕ: видеокарта недоступна (в venv стоит CPU-версия torch). Считаю на процессоре. "
              "Для видеокарты запусти install_cuda.bat.",
              "WARNING: no CUDA GPU available (CPU build of torch?). Using the CPU. "
              "Run install_cuda.bat to use the GPU."))
        dev = "cpu"
    else:
        dev = arg
    if dev.startswith("cuda"):
        try:
            log(t("Устройство: видеокарта ", "Device: GPU ") + torch.cuda.get_device_name(0))
        except Exception:
            log(t("Устройство: видеокарта", "Device: GPU"))
    else:
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass
        log(t("Устройство: процессор", "Device: CPU"))
    return dev


# ----------------------------------------------------------------- discovery --

PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.")
BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
LORA_MARKERS = (".lora_", ".lokr_", ".hada_", "lora_up", "lora_down", ".lora.", "lora_A", "lora_B")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None) or {}
    return hdr, meta


def file_kind(path):
    """-> dict(kind='model'|'lora'|'other', blocks=int, size=bytes). Reads only the header."""
    try:
        hdr, _ = read_header(path)
    except Exception:
        return {"kind": "other", "blocks": 0, "size": 0}
    is_lora = any(m in k for k in hdr for m in LORA_MARKERS) or any(k.endswith(".alpha") for k in hdr)
    blocks = set()
    for k in hdr:
        if "llm_adapter" in k:
            continue
        m = BLOCK_RE.search(k.replace("_blocks_", ".blocks.").replace("lora_unet_blocks_", "blocks."))
        if m:
            blocks.add(int(m.group(1)))
        else:
            m2 = re.search(r"blocks[._](\d+)[._]", k)
            if m2 and "llm_adapter" not in k:
                blocks.add(int(m2.group(1)))
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return {"kind": "lora" if is_lora else "model", "blocks": (max(blocks) + 1) if blocks else 0, "size": size}


def discover(folders, kind="model", blocks=None, recursive=True):
    """All .safetensors of a kind (and block count) in the folders. Skips half-downloaded files."""
    seen, out = set(), []
    for d in folders:
        d = Path(str(d).strip().strip('"')).expanduser()
        if not str(d) or not d.is_dir():
            continue
        it = d.rglob("*.safetensors") if recursive else d.glob("*.safetensors")
        for p in sorted(it, key=lambda x: str(x).lower()):
            key = os.path.normcase(str(p.resolve()))
            if key in seen:
                continue
            seen.add(key)
            info = file_kind(str(p))
            if kind and info["kind"] != kind:
                continue
            if blocks and info["blocks"] != blocks:
                continue
            out.append(str(p))
    return out


BASE_HINTS = ("anima29b", "anima-2.9b", "anima_2.9b", "anima-2-9b", "anima_basev", "anima-base", "anima_base",
              "anima-preview", "anima_preview")


def guess_base(paths, blocks=None):
    """Guess the official base among paths by file name (anima base / anima29B ...)."""
    cands = []
    for p in paths:
        n = Path(p).name.lower().replace(" ", "")
        score = 0
        if any(h in n for h in BASE_HINTS):
            score += 2
        if "anima" in n and ("base" in n or "official" in n or "v10" in n):
            score += 1
        if any(x in n for x in ("nova", "anidream", "wai", "merge", "lora", "expanded", "aesthetic")):
            score -= 2
        if score > 0:
            cands.append((score, p))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1]


def short(path, n=28):
    s = Path(str(path)).stem
    return s if len(s) <= n else s[:n - 1] + "…"


def unique_labels(paths, n=28):
    labels, used = [], {}
    for p in paths:
        s = short(p, n)
        k = used.get(s, 0)
        used[s] = k + 1
        labels.append(s if k == 0 else f"{s}#{k + 1}")
    return labels


def ltype(rest):
    r = rest.lower()
    if "adaln" in r or "modulation" in r:
        return "adaln"
    if "cross_attn" in r:
        return "cross_attn"
    if "self_attn" in r:
        return "self_attn"
    if "mlp" in r:
        return "mlp"
    return "other"


def reports_dir():
    d = ROOT / "reports"
    d.mkdir(exist_ok=True)
    return d


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def safe_print_setup():
    """Avoid UnicodeEncodeError in old Windows consoles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
