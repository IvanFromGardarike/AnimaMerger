# -*- coding: utf-8 -*-
"""Gradio UI for anima_merger.py."""
import html as _html
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# Local addresses must not go through a proxy/VPN (otherwise Gradio fails with 503)
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    os.environ[_k] = ",".join(x for x in [_v, "localhost", "127.0.0.1", "0.0.0.0", "::1"] if x)

import gradio as gr

from amutil import t, LANG, discover, guess_base, file_kind

VERSION = "1.1"
# Newer Gradio takes css/theme in launch() and warns when they are passed to Blocks(); older ones only accept Blocks().
import warnings
warnings.filterwarnings("ignore", message=r".*parameter in the Blocks constructor will be removed.*",
                        category=DeprecationWarning)
try:
    import inspect as _inspect
    STYLE_AT_LAUNCH = "css" in _inspect.signature(gr.Blocks.launch).parameters
except Exception:
    STYLE_AT_LAUNCH = False
ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "anima_merger.py"
MAP_DIR = ROOT / "maps"
MAP_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = ROOT / "settings.json"

DEFAULTS = {"models_dir": str(ROOT / "models"), "lora_dir": str(ROOT / "loras"), "out_dir": str(ROOT / "output"),
            "lang": "en", "device": "auto"}


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return s


SET = load_settings()
for _d in ("models_dir", "out_dir"):
    try:
        Path(SET[_d]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def write_settings():
    try:
        SETTINGS_FILE.write_text(json.dumps(SET, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def save_settings(models_dir, lora_dir, out_dir):
    SET.update(models_dir=models_dir.strip(), lora_dir=lora_dir.strip(), out_dir=out_dir.strip())
    write_settings()


_current = {"proc": None, "stop": False}
_lock = threading.Lock()


# ------------------------------------------------------------- utilities ----

def scan_files(folder, kind=None, blocks=None):
    folder = Path(folder or "").expanduser()
    if not str(folder) or not folder.is_dir():
        return []
    files = sorted(folder.rglob("*.safetensors"), key=lambda p: str(p).lower())
    out = []
    for p in files:
        if kind or blocks:
            info = file_kind(str(p))
            if kind and info["kind"] != kind:
                continue
            if blocks and info["blocks"] != blocks:
                continue
        out.append((str(p.relative_to(folder)), str(p)))
    return out


def scan_maps():
    return [(p.name, str(p)) for p in sorted(MAP_DIR.glob("*.json"))]


def stem(path):
    return Path(path).stem if path else "model"


def out_dir():
    d = Path(SET.get("out_dir") or DEFAULTS["out_dir"]).expanduser()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise gr.Error(t(f"Не могу создать папку вывода {d}: {e}", f"Can't create the output folder {d}: {e}"))
    return d


def out_path(name, default, folder=None, ext=".safetensors"):
    name = (name or "").strip() or default
    if not name.lower().endswith(ext):
        name += ext
    p = Path(name)
    return str(p if p.is_absolute() else (folder or out_dir()) / p)


def parse_ms(text):
    vals = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip().replace(" ", "")
        if part:
            vals.append(float(part))
    if not vals:
        raise gr.Error(t("Укажи M (например 0.5 или 0.3,0.5,0.7).", "Set M (e.g. 0.5 or 0.3,0.5,0.7)."))
    return vals


def fmt_m(m):
    return f"{m:g}".replace(".", "p").replace("-", "neg")


def need(*pairs):
    for value, label in pairs:
        if not value:
            raise gr.Error(t("Не выбрано: ", "Not selected: ") + label)


def strip_ext(base):
    return base[:-12] if base.lower().endswith(".safetensors") else base


def mult_args(m, m_adaln, m_nonblock, block_m, deep=""):
    a = ["--m", str(m)]
    if deep and deep.strip():
        a += ["--deep", " ".join(deep.replace("{M}", f"{m:g}").split())]
    if m_adaln is not None and str(m_adaln) != "":
        a += ["--m-adaln", str(m_adaln)]
    if m_nonblock is not None and str(m_nonblock) != "":
        a += ["--m-nonblock", str(m_nonblock)]
    if block_m and block_m.strip():
        a += ["--block-m", block_m.strip()]
    return a


def expert_args(out_pat, adaln_pat):
    a = []
    if out_pat and out_pat.strip():
        a += ["--out-patterns", *out_pat.split()]
    if adaln_pat and adaln_pat.strip():
        a += ["--adaln-patterns", *adaln_pat.split()]
    return a


def dtype_args(dtype):
    return [] if dtype in (None, "", "same") else ["--save-dtype", dtype]


# --------------------------------------------------------------- running ----

PROG_RE = re.compile(r"^@@PROGRESS ([\d.]+) ?(.*)$")
REP_RE = re.compile(r"^@@REPORT (.+)$")


def prog_html(frac=None, label="", state="run"):
    if frac is None:
        return ""
    pct = max(0.0, min(1.0, frac)) * 100
    col = {"run": "#2a78d6", "ok": "#0ca30c", "err": "#d03b3b"}[state]
    return (f'<div style="margin:4px 0 2px;font-size:13px;opacity:.85">{_html.escape(label)} — {pct:.0f}%</div>'
            f'<div style="height:10px;border-radius:6px;background:rgba(128,128,128,.22);overflow:hidden">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{col};transition:width .3s"></div></div>')


def run_jobs(jobs, after=None):
    """jobs: list of argument lists. Yields (log, progress_html, report_path)."""
    with _lock:
        if _current["proc"] is not None:
            yield t("Уже идёт другая операция. Дождись окончания или нажми «Стоп».",
                    "Another operation is running. Wait for it or press Stop."), "", None
            return
        _current["stop"] = False
    log = []
    ok = True
    report = None
    pg = ""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", ANIMA_PROGRESS="1",
               ANIMA_LANG=SET.get("lang", "en"))
    try:
        for n, args in enumerate(jobs, 1):
            if _current["stop"]:
                ok = False
                break
            cmd = [sys.executable, "-u", str(SCRIPT), *args]
            head = f"=== [{n}/{len(jobs)}] " if len(jobs) > 1 else "=== "
            log.append(head + " ".join(f'"{c}"' if " " in c else c for c in cmd[2:]))
            yield "\n".join(log), pg, report
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    cwd=str(ROOT), env=env, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            _current["proc"] = proc
            step_label = f"[{n}/{len(jobs)}] " if len(jobs) > 1 else ""
            for line in proc.stdout:
                line = line.rstrip("\n")
                m = PROG_RE.match(line)
                if m:
                    pg = prog_html(float(m.group(1)), step_label + m.group(2))
                    yield "\n".join(log), pg, report
                    continue
                m = REP_RE.match(line)
                if m:
                    report = m.group(1).strip()
                    continue
                log.append(line)
                if len(log) > 3000:
                    log = log[-3000:]
                yield "\n".join(log), pg, report
            proc.wait()
            _current["proc"] = None
            if _current["stop"]:
                log.append("\n■ " + t("Остановлено пользователем.", "Stopped by the user."))
                pg = prog_html(0, t("Остановлено", "Stopped"), "err")
                ok = False
            elif proc.returncode != 0:
                log.append("\n✖ " + t(f"Ошибка (код {proc.returncode}).", f"Error (code {proc.returncode})."))
                pg = prog_html(1, t("Ошибка", "Error"), "err")
                ok = False
                yield "\n".join(log), pg, report
                break
            else:
                log.append("✔ " + t("Готово.", "Done.") + "\n")
                pg = prog_html(1, step_label + t("Готово", "Done"), "ok")
            yield "\n".join(log), pg, report
        if after:
            after(ok, log)
            yield "\n".join(log), pg, report
    finally:
        p = _current["proc"]
        if p is not None and p.poll() is None:
            p.kill()
        _current["proc"] = None


def report_view(path):
    if not path or not Path(path).exists():
        return ""
    s = Path(path).read_text(encoding="utf-8")
    m = re.search(r"<body>(.*)</body>", s, re.S)
    return m.group(1) if m else s


def plain(jobs, after=None):
    for lg, pg, rep in run_jobs(jobs, after):
        yield lg, pg


def with_report(jobs):
    last = None
    for lg, pg, rep in run_jobs(jobs):
        if rep and rep != last:
            last = rep
            yield lg, pg, report_view(rep), rep
        else:
            yield lg, pg, gr.update(), gr.update()


def stop():
    _current["stop"] = True
    p = _current["proc"]
    if p is not None and p.poll() is None:
        p.kill()
    return gr.update()


def preset_args(preset, inserted, what="spec"):
    if inserted.strip():
        return ["--inserted", inserted.strip()]
    if preset == "official":
        return ["--preset", "official"]
    if preset == "even":
        return ["--preset", "even"]
    raise gr.Error(t(f"Для {what} выбери пресет или впиши позиции вставки.",
                     f"For {what}, choose a preset or enter the inserted positions."))


# -------------------------------------------------------------- handlers -----

def do_info(model, out_pat, adaln_pat):
    need((model, t("модель", "model")))
    yield from plain([["info", "--model", model, *expert_args(out_pat, adaln_pat)]])


def do_merge(a, b, c, ms, m_adaln, m_nonblock, block_m, name, dtype, dry, out_pat, adaln_pat, deep,
             mode, density, seed):
    need((a, "A"), (b, "B"), (c, "C"))
    mode = (mode or "add").split()[0]
    jobs = []
    for m in parse_ms(ms):
        args = ["merge", "--a", a, "--b", b, "--c", c, *mult_args(m, m_adaln, m_nonblock, block_m, deep),
                "--mode", mode, "--density", str(density), "--seed", str(int(seed or 0))]
        if dry:
            args.append("--dry-run")
        else:
            base = strip_ext((name or "").strip() or f"{stem(b)}__on__{stem(a)}")
            suffix = f"_{mode}" if mode != "add" and not (name or "").strip() else ""
            args += ["--out", out_path(f"{base}{suffix}_M{fmt_m(m)}", "")]
        jobs.append(args + dtype_args(dtype) + expert_args(out_pat, adaln_pat))
    yield from plain(jobs)


def do_map(small, large, source, inserted, preset, keep, name, sample, out_pat, adaln_pat):
    need((small, t("база 2B", "2B base")), (large, t("база 2.9B", "2.9B base")))
    args = ["map", "--small", small, "--large", large, "--source", source, "--keep", keep,
            "--sample", str(int(sample))]
    if source in ("spec", "compare"):
        args += preset_args(preset, inserted, source)
    args += ["--out", out_path(name, "anima_map", MAP_DIR, ".json")]
    yield from plain([args + expert_args(out_pat, adaln_pat)])


def do_cross(target, ft, ft_base, map_source, ref_base, inserted, preset, copy_from, map_file,
             save_map, m_ins, ins_src, fold, ms, m_adaln, m_nonblock, block_m, name, dtype, dry,
             sample, out_pat, adaln_pat, deep=""):
    need((target, "target"), (ft, t("файнтюн", "fine-tune")), (ft_base, t("база файнтюна", "fine-tune base")))
    common = ["--target", target, "--ft", ft, "--ft-base", ft_base, "--map-source", map_source,
              "--m-inserted", str(m_ins), "--inserted-source", ins_src,
              "--fold-inserted", str(fold), "--sample", str(int(sample))]
    if map_source == "auto" and ref_base:
        common += ["--ref-base", ref_base]
    if map_source == "spec":
        common += preset_args(preset, inserted) + ["--copy-from", copy_from]
    if map_source == "file":
        need((map_file, t("файл карты", "map file")))
        common += ["--map-file", map_file]
    jobs = []
    for k, m in enumerate(parse_ms(ms)):
        args = ["cross", *common, *mult_args(m, m_adaln, m_nonblock, block_m, deep)]
        if save_map and k == 0 and map_source != "file":
            args += ["--save-map", out_path(f"map_{stem(ft_base)}_{stem(target)}", "", MAP_DIR, ".json")]
        if dry:
            args.append("--dry-run")
        else:
            base = strip_ext((name or "").strip() or f"{stem(ft)}__ported__{stem(target)}")
            args += ["--out", out_path(f"{base}_M{fmt_m(m)}", "")]
        jobs.append(args + dtype_args(dtype) + expert_args(out_pat, adaln_pat))
    yield from plain(jobs)


def lora_specs(rows):
    """LoRA table rows -> ['path:strength', ...]"""
    specs = []
    data = rows.get("data", rows) if isinstance(rows, dict) else rows
    if hasattr(data, "values"):
        data = data.values.tolist()
    for r in data or []:
        if not r or not r[0] or not str(r[0]).strip():
            continue
        path = str(r[0]).strip()
        p = Path(path)
        if not p.is_absolute():
            cand = Path(SET.get("lora_dir", "")) / p
            path = str(cand) if cand.exists() else path
        try:
            s = float(r[1]) if len(r) > 1 and str(r[1]).strip() != "" else 1.0
        except ValueError:
            raise gr.Error(t(f"Сила LoRA должна быть числом: {r[1]}", f"LoRA strength must be a number: {r[1]}"))
        specs.append(f"{path}:{s:g}")
    return specs


def add_lora_row(rows, lora, strength):
    data = rows.get("data", rows) if isinstance(rows, dict) else rows
    if hasattr(data, "values"):
        data = data.values.tolist()
    data = [list(r) for r in (data or []) if r and str(r[0]).strip()]
    if lora:
        data.append([lora, float(strength)])
    return data or [["", 1.0]]


def do_bake(model, rows, deep, cap, m_ins, name, dtype, dry, grid):
    need((model, t("модель", "model")))
    specs = lora_specs(rows)
    if not specs:
        raise gr.Error(t("Добавь хотя бы одну LoRA.", "Add at least one LoRA."))
    mults = parse_ms(grid) if grid and grid.strip() else [None]
    jobs = []
    for g in mults:
        sp = specs if g is None else [f"{s.rsplit(':', 1)[0]}:{float(s.rsplit(':', 1)[1]) * g:g}" for s in specs]
        args = ["bake", "--model", model]
        for s in sp:
            args += ["--lora", s]
        if deep and deep.strip():
            args += ["--deep", " ".join(deep.split())]
        if cap:
            args += ["--cap", str(cap)]
        if m_ins:
            args += ["--m-inserted", str(m_ins)]
        if dry:
            args.append("--dry-run")
        else:
            base = strip_ext((name or "").strip() or
                             f"{stem(model)}+{'+'.join(stem(s.rsplit(':', 1)[0]) for s in specs)}")
            if g is not None:
                base += f"_x{fmt_m(g)}"
            args += ["--out", out_path(base[:180], "")]
        jobs.append(args + dtype_args(dtype))
    yield from plain(jobs)


def do_extract(tuned, base, rank, rank_adaln, deep, nonblock, clamp, device, name, dtype, dry):
    need((tuned, t("модель со стилем", "model with the style")), (base, t("база", "base")))
    args = ["extract", "--tuned", tuned, "--base", base, "--rank", str(int(rank)), "--device", device,
            "--save-dtype", dtype if dtype in ("bf16", "fp16", "fp32") else "bf16"]
    if rank_adaln:
        args += ["--rank-adaln", str(int(rank_adaln))]
    if deep and deep.strip():
        args += ["--deep", " ".join(deep.split())]
    if nonblock:
        args.append("--include-nonblock")
    if clamp:
        args += ["--clamp", str(clamp)]
    if dry:
        args.append("--dry-run")
    else:
        folder = Path(SET.get("lora_dir") or "") if SET.get("lora_dir") else None
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
        args += ["--out", out_path((name or "").strip() or f"{stem(tuned)}-minus-{stem(base)}_r{int(rank)}",
                                   "", folder)]
    yield from plain([args])


def do_lora_merge(rows, deep, mode, rank, device, name, dtype, dry):
    specs = lora_specs(rows)
    if len(specs) < 2:
        raise gr.Error(t("Добавь хотя бы две LoRA.", "Add at least two LoRAs."))
    args = ["lora-merge"]
    for s in specs:
        args += ["--lora", s]
    args += ["--mode", mode, "--rank", str(int(rank)), "--device", device, "--save-dtype", dtype]
    if deep and deep.strip():
        args += ["--deep", " ".join(deep.split())]
    if dry:
        args.append("--dry-run")
    else:
        folder = Path(SET.get("lora_dir") or "") if SET.get("lora_dir") else None
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
        base = strip_ext((name or "").strip() or "+".join(stem(s.rsplit(":", 1)[0]) for s in specs))
        args += ["--out", out_path(base[:180], "", folder)]
    yield from plain([args])


def do_convert(files, folder, to, m_ins, out_folder):
    items = list(files or [])
    if folder and folder.strip():
        items.append(folder.strip().strip('"'))
    if not items:
        raise gr.Error(t("Выбери LoRA или укажи папку.", "Pick LoRA files or a folder."))
    args = ["lora-convert", "--lora", *items, "--to", to, "--m-inserted", str(m_ins)]
    if out_folder and out_folder.strip():
        args += ["--out-dir", out_folder.strip().strip('"')]
    yield from plain([args])


# --- analyzers

def models_list(sel):
    return [x for x in (sel or []) if x]


def do_parent(target, cands, base, device, exact=False):
    need((target, t("цель", "target")))
    cands = models_list(cands)
    if not cands:
        cands = [p for p in discover([SET["models_dir"]], "model", file_kind(target)["blocks"])]
    args = ["parent", "--target", target, "--cands", *cands, "--device", device] + (["--exact"] if exact else [])
    if base:
        args += ["--base", base]
    yield from with_report([args])


def do_family(models, ref, device, exact=False):
    need((ref, t("эталон", "reference")))
    ms = models_list(models)
    if len(ms) < 2:
        raise gr.Error(t("Выбери хотя бы 2 модели.", "Pick at least 2 models."))
    yield from with_report([["family", "--models", *ms, "--ref", ref, "--device", device] + (["--exact"] if exact else [])])


def do_genome(models, base, device, exact=False):
    need((base, t("база", "base")))
    ms = models_list(models)
    if len(ms) < 2:
        raise gr.Error(t("Выбери хотя бы 2 модели.", "Pick at least 2 models."))
    yield from with_report([["genome", "--models", *ms, "--base", base, "--device", device] + (["--exact"] if exact else [])])


def do_blocks(models, ref, base2b, device, exact=False):
    need((ref, t("эталон", "reference")))
    ms = models_list(models)
    if not ms:
        raise gr.Error(t("Выбери модели.", "Pick models."))
    args = ["blocks", "--models", *ms, "--ref", ref, "--device", device] + (["--exact"] if exact else [])
    if base2b:
        args += ["--base2b", base2b]
    yield from with_report([args])


PROFILES_DIR = ROOT / "profiles"


def profiles_summary():
    files = list(PROFILES_DIR.glob("*.json")) if PROFILES_DIR.exists() else []
    size = sum(f.stat().st_size for f in PROFILES_DIR.glob("*") if f.is_file()) if PROFILES_DIR.exists() else 0
    mb = size / 1024 / 1024
    return t(f"Профилей: **{len(files)}**, занято **{mb:.1f} МБ**. Папка: `{PROFILES_DIR}`",
             f"Profiles: **{len(files)}**, using **{mb:.1f} MB**. Folder: `{PROFILES_DIR}`")


def do_profiles_build(folder, rebuild):
    folder = (folder or "").strip() or SET["models_dir"]
    if not os.path.isdir(folder):
        raise gr.Error(t(f"Нет такой папки: {folder}", f"No such folder: {folder}"))
    yield from plain([["profile", "--models", folder] + (["--rebuild"] if rebuild else [])])


def do_profiles_clear(ok):
    if not ok:
        raise gr.Error(t("Поставь галочку «Да, удалить все профили».", "Tick 'Yes, delete all profiles' first."))
    n = 0
    if PROFILES_DIR.exists():
        for f in PROFILES_DIR.iterdir():
            if f.is_file() and f.suffix in (".json", ".npz", ".part"):
                f.unlink()
                n += 1
    gr.Info(t(f"Удалено файлов: {n}. Профили построятся заново при следующем анализе.",
              f"Files deleted: {n}. Profiles will be rebuilt on the next analysis."))
    return profiles_summary(), False


def do_meta(files, sha):
    fs = models_list(files)
    if not fs:
        raise gr.Error(t("Выбери файлы.", "Pick files."))
    yield from with_report([["meta", "--files", *fs] + (["--sha256"] if sha else [])])


def do_lora_info(files, folder, steps, model, device):
    items = models_list(files)
    if folder and folder.strip():
        items.append(folder.strip().strip('"'))
    if not items:
        raise gr.Error(t("Выбери LoRA или укажи папку с чекпоинтами.", "Pick LoRA files or a folder with checkpoints."))
    args = ["lora-info", "--loras", *items, "--device", device]
    if steps and steps.strip():
        args += ["--steps", steps.strip()]
    if model:
        args += ["--model", model]
    yield from with_report([args])


# ---------------------------------------------------------------- recipe --

RECIPE_HELP = t("""**Рецепт** — цепочка шагов, каждый в своей строке. `Имя = команда ключ=значение ...`
Результат шага сохраняется как `Имя.safetensors` в папку вывода, на него можно сослаться дальше через `$Имя`.
Имена моделей без пути ищутся в папке моделей, LoRA — в папке LoRA. Значения с пробелами — в кавычках. `#` — комментарий.

```
# 1. Flat: в слоях 16-27 MLP вернуть модель к базе на 50%
Flat = merge a=MyMerge.safetensors b=anima_base.safetensors c=MyMerge.safetensors m=0.5 deep="0, L16-L27:mlp.layer1 mlp.layer2:0.5" adaln=0 nonblock=0
# 2. Только самые сильные 20% дельты другого файнтюна
FlatPlus = merge a=$Flat b=other_finetune.safetensors c=anima_base.safetensors m=0.3 mode=topk density=0.2
# 3. Вшить LoRA стиля
Final = bake model=$FlatPlus lora=my_style.safetensors:0.6 lora=my_char.safetensors:0.8 cap=0.06
```
Команды: `merge` (a b c m mode density seed deep adaln nonblock block_m dtype), `bake` (model lora… deep cap m_inserted dtype),
`extract` (tuned base rank rank_adaln deep dtype), `cross` (target ft ft_base m preset m_inserted deep dtype).""",
"""**Recipe** — a chain of steps, one per line: `Name = command key=value ...`
Each step is saved as `Name.safetensors` in the output folder and can be referenced later as `$Name`.
Model names without a path are looked up in the models folder, LoRAs in the LoRA folder. Quote values with spaces. `#` starts a comment.

```
# 1. Flat: pull MLP layers 16-27 halfway back to the base
Flat = merge a=MyMerge.safetensors b=anima_base.safetensors c=MyMerge.safetensors m=0.5 deep="0, L16-L27:mlp.layer1 mlp.layer2:0.5" adaln=0 nonblock=0
# 2. Only the strongest 20% of another fine-tune's delta
FlatPlus = merge a=$Flat b=other_finetune.safetensors c=anima_base.safetensors m=0.3 mode=topk density=0.2
# 3. Bake a style LoRA
Final = bake model=$FlatPlus lora=my_style.safetensors:0.6 lora=my_char.safetensors:0.8 cap=0.06
```
Commands: `merge` (a b c m mode density seed deep adaln nonblock block_m dtype), `bake` (model lora… deep cap m_inserted dtype),
`extract` (tuned base rank rank_adaln deep dtype), `cross` (target ft ft_base m preset m_inserted deep dtype).""")

KEY_MAP = {"adaln": "--m-adaln", "nonblock": "--m-nonblock", "block_m": "--block-m", "dtype": "--save-dtype",
           "ft_base": "--ft-base", "m_inserted": "--m-inserted", "rank_adaln": "--rank-adaln",
           "map_source": "--map-source", "map_file": "--map-file", "ref_base": "--ref-base",
           "inserted_source": "--inserted-source", "fold_inserted": "--fold-inserted"}
MODEL_KEYS = {"a", "b", "c", "model", "tuned", "base", "target", "ft", "ft_base", "ref_base"}
TOKEN_RE = re.compile(r'([A-Za-z_][\w-]*)=("([^"]*)"|\S+)')


def find_in_dirs(val, dirs, ln, what):
    """Model/LoRA from a recipe: path relative to a folder, otherwise found by file name in its subfolders."""
    names = [val] if val.lower().endswith(".safetensors") else [val, val + ".safetensors"]
    dirs = [Path(d).expanduser() for d in dirs if d and Path(d).expanduser().is_dir()]
    for d in dirs:
        for n in names:
            if (d / n).is_file():
                return str(d / n)
    want = {Path(n).name.lower() for n in names}
    sub = {Path(n).parent.as_posix().lower() for n in names}
    for d in dirs:
        hits = [p for p in d.rglob("*.safetensors") if p.name.lower() in want
                and (sub == {"."} or any(("/" + p.relative_to(d).parent.as_posix().lower()).endswith("/" + x) for x in sub))]
        if len(hits) == 1:
            return str(hits[0])
        if len(hits) > 1:
            lst = "\n".join("  " + str(p.relative_to(d)) for p in sorted(hits, key=lambda p: str(p).lower()))
            raise gr.Error(t(f"Строка {ln}: {what} {val} есть в нескольких подпапках {d}, укажи путь:\n{lst}",
                             f"Line {ln}: {what} {val} is in several subfolders of {d}, give the path:\n{lst}"))
    where = ", ".join(str(d) for d in dirs) or "—"
    import difflib
    allp = {}
    for d in dirs:
        for p in d.rglob("*.safetensors"):
            allp.setdefault(p.stem.lower(), str(p.relative_to(d)))
    key = Path(names[0]).stem.lower()
    near = difflib.get_close_matches(key, list(allp), n=5, cutoff=0.4)
    words = [w for w in re.split(r"[^a-z0-9]+", key) if len(w) > 2]
    near += [k for k in sorted(allp) if k not in near and words and sum(w in k for w in words) >= max(1, len(words) - 1)][:8 - len(near)]
    hint = ("\n" + t("Похожие:", "Similar:") + "\n" + "\n".join("  " + allp[k] for k in near)) if near else ""
    raise gr.Error(t(f"Строка {ln}: не найдено: {what} {val} в {where} (и подпапках){hint}",
                     f"Line {ln}: {what} {val} not found in {where} (and subfolders){hint}"))


def parse_recipe(text, used=None):
    steps, outputs = [], {}
    used = used if used is not None else set()
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if not line:
            continue
        m = re.match(r"^([A-Za-z_][\w-]*)\s*=\s*(merge|bake|extract|cross)\b(.*)$", line)
        if not m:
            raise gr.Error(t(f"Строка {ln}: нужно «Имя = команда ключ=значение ...»",
                             f"Line {ln}: expected 'Name = command key=value ...'"))
        name, cmd, rest = m.group(1), m.group(2), m.group(3)
        args = [cmd]
        has_preset = False
        for tm in TOKEN_RE.finditer(rest):
            key, val = tm.group(1).lower(), tm.group(3) if tm.group(3) is not None else tm.group(2)
            if key in ("out", "name"):
                continue
            if val.startswith("$"):
                ref = val[1:]
                if ref not in outputs:
                    raise gr.Error(t(f"Строка {ln}: шаг ${ref} ещё не определён выше.",
                                     f"Line {ln}: step ${ref} is not defined above."))
                used.add(ref)
                val = outputs[ref]
            elif key in MODEL_KEYS and not Path(val).is_absolute():
                val = find_in_dirs(val, [SET["models_dir"], out_dir()], ln, t("модель", "model"))
            elif key == "lora":
                p, _, s = val.rpartition(":")
                try:
                    float(s)
                except ValueError:
                    p, s = val, "1"
                if p.startswith("$"):
                    used.add(p[1:])
                    p = outputs.get(p[1:], p)
                elif not Path(p).is_absolute():
                    p = find_in_dirs(p, [SET["lora_dir"]], ln, "LoRA")
                val = f"{p}:{s}"
            if key == "preset":
                has_preset = True
                val = "official" if val.startswith("off") else val
            args += [KEY_MAP.get(key, "--" + key.replace("_", "-")), val]
        if cmd == "cross" and has_preset and "--map-source" not in args:
            args += ["--map-source", "spec"]
        out = out_path(name, "", folder=(Path(SET["lora_dir"]) if cmd == "extract" else None))
        outputs[name] = out
        steps.append((name, args + ["--out", out], out))
    if not steps:
        raise gr.Error(t("Рецепт пустой.", "The recipe is empty."))
    return steps


def do_recipe(text, cleanup):
    used = set()
    steps = parse_recipe(text, used)

    def after(ok, log):
        if not ok:
            log.append(t("Рецепт остановлен; промежуточные файлы оставлены.",
                         "The recipe stopped; intermediate files were kept."))
            return
        if cleanup:
            for name, _, out in steps:
                if name not in used:
                    continue  # final results (not referenced by anyone) are kept
                try:
                    Path(out).unlink()
                    log.append(t("Удалён промежуточный файл: ", "Removed intermediate file: ") + out)
                except Exception as e:
                    log.append(t(f"Не удалось удалить {out}: {e}", f"Could not remove {out}: {e}"))
        finals = [out for name, _, out in steps if name not in used]
        log.append("\n" + t("Итог: ", "Result: ") + "\n      ".join(finals))

    yield from plain([a for _, a, _ in steps], after)


def check_recipe(text):
    steps = parse_recipe(text)
    return "\n".join(f"{n}. {name}: " + " ".join(f'"{x}"' if " " in x else x for x in args)
                     for n, (name, args, _) in enumerate(steps, 1))


# ----------------------------------------------------------------- UI -----

CSS = """
#log textarea {font-family: Consolas, 'Courier New', monospace; font-size: 12.5px;}
#recipe textarea {font-family: Consolas, 'Courier New', monospace; font-size: 13px;}
.small-note {opacity: .75; font-size: .9em}
#lang {max-width: 200px}
"""

DTYPES_UI = [(t("как у исходной модели", "same as the source"), "same"), ("bf16", "bf16"), ("fp16", "fp16"),
             ("fp32", "fp32")]
MODES_UI = [(t("add — обычная дельта", "add — plain delta"), "add"),
            (t("topk — только самые сильные изменения", "topk — only the strongest changes"), "topk"),
            (t("dare — случайное прореживание", "dare — random drop"), "dare"),
            (t("ortho — без части, параллельной весам A", "ortho — without the part parallel to A's weights"), "ortho"),
            (t("ortho_delta — без того, что A уже взяла от C (без двойной дозы)",
               "ortho_delta — without what A already took from C (no double dose)"), "ortho_delta"),
            (t("normdir — без «выцветания»", "normdir — no 'fading'"), "normdir")]
PRESETS_UI = [(t("official (28→40)", "official (28→40)"), "official"), (t("even (догадка)", "even (guess)"), "even"),
              (t("свои позиции", "custom positions"), "custom")]
DEEP_INFO = t("Как в рецептах Forge Neo: блоки, подстроки имён слоёв через пробел, значение. "
              "Число в начале = M по умолчанию. {M} подставляет значение из поля M. "
              "adapter:*:0 — правило для llm_adapter. ^cross_attn — только слои, чьё имя начинается с cross_attn "
              "(без adaln_modulation_cross_attn).",
              "As in Forge Neo recipes: blocks, layer-name substrings separated by spaces, value. "
              "A number first = default M. {M} inserts the value from the M field. "
              "adapter:*:0 — a rule for llm_adapter. ^cross_attn — only layers whose name starts with cross_attn "
              "(not adaln_modulation_cross_attn).")
DEVICES = [(t("авто (видеокарта, если есть)", "auto (GPU if available)"), "auto"), ("cuda", "cuda"), ("cpu", "cpu")]


def switch_lang(v):
    if v == SET.get("lang"):
        return gr.update()
    SET["lang"] = v
    write_settings()

    def restart():
        if os.environ.get("ANIMA_RUNNER") == "1":
            os._exit(3)  # run.bat / run.sh restart the app on exit code 3
        env = dict(os.environ, ANIMA_NO_BROWSER="1")
        flags = 0x01000000 if os.name == "nt" else 0  # CREATE_BREAKAWAY_FROM_JOB: the new app outlives this one
        subprocess.Popen([sys.executable] + sys.argv, cwd=str(ROOT), env=env, creationflags=flags)
        os._exit(0)

    threading.Timer(0.8, restart).start()
    return gr.update()


def build():
    with gr.Blocks(title="Anima Merger", **({} if STYLE_AT_LAUNCH else {"css": CSS, "theme": gr.themes.Soft()})) as ui:
        with gr.Row():
            gr.Markdown(f"## Anima Merger {VERSION} — " + t("мердж, LoRA, перенос 2B ↔ 2.9B, анализ",
                                                             "merging, LoRA, 2B ↔ 2.9B porting, analysis"))
            lang = gr.Radio([("English", "en"), ("Русский", "ru")], value=SET.get("lang", "en"),
                            label="Language / Язык", elem_id="lang", scale=0)
        with gr.Row():
            models_dir = gr.Textbox(label=t("Папка с моделями", "Models folder"), value=SET["models_dir"], scale=4)
            lora_dir = gr.Textbox(label=t("Папка с LoRA", "LoRA folder"), value=SET["lora_dir"], scale=4)
            refresh = gr.Button(t("🔄 Обновить списки", "🔄 Refresh lists"), scale=1)
        out_dir_tb = gr.Textbox(label=t("Папка вывода", "Output folder"), value=SET["out_dir"],
                                info=t("Куда сохранять результаты. Если она на том же диске, что и модели, "
                                       "файлы не придётся таскать между дисками. Запоминается.",
                                       "Where results are saved. On the same drive as the models, big files don't "
                                       "have to be copied between drives. Remembered."))

        dds, ldds, mdds, mlds = [], [], [], []

        def model_dd(label, info=None):
            d = gr.Dropdown(label=label, info=info, choices=scan_files(SET["models_dir"]), value=None,
                            allow_custom_value=True, filterable=True)
            dds.append(d)
            return d

        def models_multi(label, info=None):
            d = gr.Dropdown(label=label, info=info, choices=scan_files(SET["models_dir"]), multiselect=True,
                            value=[], allow_custom_value=True, filterable=True)
            mdds.append(d)
            return d

        def pick_buttons(target):
            with gr.Row():
                b28 = gr.Button(t("Все 2B (28 блоков)", "All 2B (28 blocks)"), size="sm")
                b40 = gr.Button(t("Все 2.9B (40 блоков)", "All 2.9B (40 blocks)"), size="sm")
                bclr = gr.Button(t("Очистить", "Clear"), size="sm")
            b28.click(lambda d: discover([d], "model", 28), models_dir, target)
            b40.click(lambda d: discover([d], "model", 40), models_dir, target)
            bclr.click(lambda: [], None, target)

        def guess_btn(models_sel, target, blocks_from=None):
            def g(sel, d):
                paths = list(sel or [])
                nb = file_kind(paths[0])["blocks"] if paths else None
                cands = discover([d], "model", nb) if nb else discover([d], "model")
                return guess_base(cands) or gr.update()
            b = gr.Button(t("Найти базу по имени", "Guess the base by name"), size="sm")
            b.click(g, [models_sel, models_dir], target)

        with gr.Tabs():
            # ---------------- merge
            with gr.Tab(t("Мердж / замена базы", "Merge / rebase")):
                gr.Markdown(t("**A + (B − C) × M** — модели одной архитектуры.  "
                              "Для замены базы: A = новая база, B = файнтюн, C = старая база, M = 1.",
                              "**A + (B − C) × M** — models of the same architecture.  "
                              "To rebase: A = new base, B = fine-tune, C = old base, M = 1."),
                            elem_classes="small-note")
                with gr.Row():
                    mA = model_dd(t("A — к чему добавляем", "A — add to"), t("для rebase: новая база", "rebase: the new base"))
                    mB = model_dd(t("B — откуда берём дельту", "B — take the delta from"), t("файнтюн", "fine-tune"))
                    mC = model_dd(t("C — база, от которой обучалась B", "C — the base B was trained from"))
                with gr.Row():
                    m_ms = gr.Textbox(label=t("M (можно несколько: 0.3, 0.5, 0.7)", "M (several allowed: 0.3, 0.5, 0.7)"),
                                      value="1.0")
                    m_name = gr.Textbox(label=t("Имя выходного файла (пусто = автоматически)", "Output file name (empty = automatic)"))
                    m_dtype = gr.Dropdown(DTYPES_UI, value="same", label=t("Тип сохранения", "Save as"))
                with gr.Row():
                    m_mode = gr.Dropdown(MODES_UI, value="add", label=t("Как добавлять дельту", "How to add the delta"),
                                         info=t("topk — тащит концепты без «шума» донора; ortho — меньше ломает стиль A",
                                                "topk — takes concepts without the donor's 'noise'; ortho — breaks A's style less"))
                    m_dens = gr.Slider(0.02, 1, value=0.2, step=0.01, label=t("topk/dare: доля дельты", "topk/dare: delta share"),
                                       info=t("0.1–0.3 — обычно", "0.1–0.3 is typical"))
                    m_seed = gr.Number(label=t("dare: сид", "dare: seed"), value=0, precision=0)
                with gr.Accordion(t("Дополнительные множители", "More multipliers"), open=False):
                    with gr.Row():
                        m_adaln = gr.Number(label=t("M для adaLN (пусто = как M, 0 = оставить от A)",
                                                    "M for adaLN (empty = as M, 0 = keep A)"), value=None)
                        m_nonb = gr.Number(label=t("M для не-блочных весов (пусто = как M)", "M for non-block weights (empty = as M)"),
                                           value=None)
                    m_blk = gr.Textbox(label=t("Поблочные коэффициенты (× M)", "Per-block factors (× M)"),
                                       placeholder="0-9:0.5,30-39:1.2")
                    m_deep = gr.Textbox(label=t("Точные правила по блокам и слоям", "Exact rules per block and layer"), lines=2,
                                        placeholder="0, L16-L27:mlp.layer1 mlp.layer2:{M}, adapter:*:0", info=DEEP_INFO)
                with gr.Row():
                    m_dry = gr.Checkbox(label=t("Пробный прогон (не сохранять)", "Dry run (don't save)"))
                    m_go = gr.Button(t("▶ Смержить", "▶ Merge"), variant="primary")

            # ---------------- bake
            with gr.Tab(t("Вшить LoRA", "Bake LoRA")):
                gr.Markdown(t("Вшивает одну или несколько LoRA / LoHa / LoKr / DoRA в модель. Понимает форматы kohya, "
                              "diffusion-pipe, ComfyUI. LoRA от 2B на модели 2.9B (и наоборот) раскладывается "
                              "по официальной схеме расширения.",
                              "Bakes one or more LoRA / LoHa / LoKr / DoRA into a model. Understands kohya, diffusion-pipe "
                              "and ComfyUI formats. A 2B LoRA on a 2.9B model (and back) is mapped by the official "
                              "expansion layout."), elem_classes="small-note")
                bM = model_dd(t("Модель", "Model"))
                with gr.Row():
                    b_pick = gr.Dropdown(label=t("LoRA из папки", "LoRA from the folder"), choices=scan_files(SET["lora_dir"]), value=None,
                                         allow_custom_value=True, filterable=True, scale=4)
                    ldds.append(b_pick)
                    b_str = gr.Number(label=t("Сила", "Strength"), value=1.0, scale=1)
                    b_add = gr.Button(t("➕ Добавить", "➕ Add"), scale=1)
                b_rows = gr.Dataframe(headers=[t("LoRA (путь или имя в папке LoRA)", "LoRA (path or name in the LoRA folder)"),
                                               t("Сила", "Strength")],
                                      datatype=["str", "number"], value=[["", 1.0]],
                                      col_count=(2, "fixed"), interactive=True, label=t("Список LoRA", "LoRA list"))
                with gr.Row():
                    b_cap = gr.Slider(0, 0.3, value=0, step=0.01, label=t("Ограничитель изменения слоя", "Layer change limiter"),
                                      info=t("0 = выкл. 0.03–0.08 — защита от «пережарки», когда вшиваешь несколько LoRA",
                                             "0 = off. 0.03–0.08 protects from over-baking when baking several LoRAs"))
                    b_mins = gr.Slider(0, 1, value=0, step=0.05,
                                       label=t("LoRA 2B → модель 2.9B: доля для вставленных блоков",
                                               "2B LoRA → 2.9B model: share for inserted blocks"),
                                       info=t("0 = не трогать вставленные блоки", "0 = leave the inserted blocks alone"))
                b_deep = gr.Textbox(label=t("Множители по блокам и слоям", "Multipliers per block and layer"), lines=1,
                                    placeholder="1, L00-L03:*:0.5, L20-L27:mlp:0, adapter:*:0",
                                    info=t("Число в начале — общий множитель (1). Удобно, чтобы LoRA персонажа не тащила чужой стиль. ",
                                           "A number first is the global multiplier (1). Handy to keep a character LoRA from dragging in its style. ")
                                    + DEEP_INFO)
                with gr.Row():
                    b_grid = gr.Textbox(label=t("Сетка общей силы (необязательно)", "Global strength grid (optional)"),
                                        placeholder="0.6, 0.8, 1.0",
                                        info=t("Умножает силу всех LoRA; сохранит по файлу на значение",
                                               "Multiplies all LoRA strengths; saves one file per value"))
                    b_name = gr.Textbox(label=t("Имя выходного файла (пусто = автоматически)", "Output file name (empty = automatic)"))
                    b_dtype = gr.Dropdown(DTYPES_UI, value="same", label=t("Тип сохранения", "Save as"))
                with gr.Row():
                    b_dry = gr.Checkbox(label=t("Пробный прогон (только проверить, что LoRA ложится)",
                                                "Dry run (only check that the LoRA fits)"))
                    b_go = gr.Button(t("▶ Вшить", "▶ Bake"), variant="primary")

            # ---------------- extract
            with gr.Tab(t("Извлечь LoRA", "Extract LoRA")):
                gr.Markdown(t("Делает LoRA из разницы двух моделей (SVD). Например, модель со стилем = твой мердж, "
                              "база = то, из чего он сделан. Результат сохраняется в папку LoRA.",
                              "Makes a LoRA from the difference of two models (SVD). E.g. model with the style = your merge, "
                              "base = what it was made from. The result goes to the LoRA folder."),
                            elem_classes="small-note")
                with gr.Row():
                    xT = model_dd(t("Модель со стилем / концептами", "Model with the style / concepts"))
                    xB = model_dd(t("База (что вычесть)", "Base (what to subtract)"))
                with gr.Row():
                    x_rank = gr.Slider(4, 256, value=32, step=4, label=t("Ранг", "Rank"),
                                       info=t("16–32 стиль, 64–128 если разница большая", "16–32 for a style, 64–128 for a big difference"))
                    x_ra = gr.Number(label=t("Ранг для adaLN (0 = как общий)", "adaLN rank (0 = same)"), value=0, precision=0)
                    x_dev = gr.Radio(["auto", "cuda", "cpu"], value="auto", label=t("Устройство", "Device"))
                x_deep = gr.Textbox(label=t("Какие блоки/слои брать", "Which blocks/layers to take"), placeholder="1, L00-L03:*:0",
                                    info=t("Пусто = все блоки. 0 у правила = не включать эти слои в LoRA",
                                           "Empty = all blocks. 0 in a rule = leave those layers out"))
                with gr.Row():
                    x_nonb = gr.Checkbox(label=t("Включить адаптер и не-блочные слои", "Include the adapter and non-block layers"))
                    x_clamp = gr.Number(label=t("Обрезка выбросов (квантиль, 0 = выкл, напр. 0.99)",
                                                "Clip outliers (quantile, 0 = off, e.g. 0.99)"), value=0)
                    x_dtype = gr.Dropdown(["bf16", "fp16", "fp32"], value="bf16", label=t("Тип сохранения", "Save as"))
                x_name = gr.Textbox(label=t("Имя файла LoRA (пусто = автоматически)", "LoRA file name (empty = automatic)"))
                with gr.Row():
                    x_dry = gr.Checkbox(label=t("Пробный прогон (оценить точность без сохранения)", "Dry run (estimate accuracy, don't save)"))
                    x_go = gr.Button(t("▶ Извлечь", "▶ Extract"), variant="primary")

            # ---------------- merge LoRAs
            with gr.Tab(t("Смешать LoRA", "Mix LoRAs")):
                gr.Markdown(t("Смешивает несколько LoRA в одну, без модели. **concat** — точно, без потерь: ранг "
                              "итоговой LoRA = сумма рангов. **svd** — сжимает смесь до заданного ранга (меньше файл, "
                              "небольшая потеря точности, покажет её в логе). Правила по блокам позволяют, например, "
                              "взять композицию от одной LoRA, а линию и фактуру от другой. Результат — в папку LoRA.",
                              "Mixes several LoRAs into one, no model needed. **concat** — exact, lossless: the result's "
                              "rank = sum of ranks. **svd** — compresses the mix to a chosen rank (smaller file, a small "
                              "accuracy loss shown in the log). Block rules let you e.g. take composition from one LoRA "
                              "and line and texture from another. The result goes to the LoRA folder."),
                            elem_classes="small-note")
                with gr.Row():
                    j_pick = gr.Dropdown(label=t("LoRA из папки", "LoRA from the folder"), choices=scan_files(SET["lora_dir"]),
                                         value=None, allow_custom_value=True, filterable=True, scale=4)
                    ldds.append(j_pick)
                    j_str = gr.Number(label=t("Сила", "Strength"), value=0.5, scale=1)
                    j_add = gr.Button(t("➕ Добавить", "➕ Add"), scale=1)
                j_rows = gr.Dataframe(headers=[t("LoRA (путь или имя в папке LoRA)", "LoRA (path or name in the LoRA folder)"),
                                               t("Сила", "Strength")],
                                      datatype=["str", "number"], value=[["", 0.5]],
                                      col_count=(2, "fixed"), interactive=True, label=t("Список LoRA", "LoRA list"))
                j_deep = gr.Textbox(label=t("Множители по блокам и слоям (для всех LoRA)", "Multipliers per block and layer (all LoRAs)"),
                                    placeholder="1, L00-L13:*:0.5, adapter:*:0", info=DEEP_INFO)
                with gr.Row():
                    j_mode = gr.Radio([(t("авто (concat, если можно)", "auto (concat if possible)"), "auto"),
                                       ("concat", "concat"), ("svd", "svd")], value="auto", label=t("Способ", "Method"))
                    j_rank = gr.Slider(4, 256, value=64, step=4, label=t("svd: ранг результата", "svd: result rank"),
                                       info=t("для двух LoRA ранга 32 — 64 без потерь", "for two rank-32 LoRAs, 64 is lossless"))
                    j_dev = gr.Radio(["auto", "cuda", "cpu"], value="auto", label=t("Устройство", "Device"))
                with gr.Row():
                    j_name = gr.Textbox(label=t("Имя файла (пусто = автоматически)", "File name (empty = automatic)"))
                    j_dtype = gr.Dropdown(["bf16", "fp16", "fp32"], value="bf16", label=t("Тип сохранения", "Save as"))
                with gr.Row():
                    j_dry = gr.Checkbox(label=t("Пробный прогон (не сохранять)", "Dry run (don't save)"))
                    j_go = gr.Button(t("▶ Смешать", "▶ Mix"), variant="primary")

            # ---------------- convert LoRA
            with gr.Tab(t("Конвертер LoRA 2B ↔ 2.9B", "LoRA converter 2B ↔ 2.9B")):
                gr.Markdown(t("Перекладывает файл LoRA на другую архитектуру по официальной схеме расширения: "
                              "блоки 2B встают на свои места в 2.9B. Получается обычная LoRA, которую можно подключать "
                              "в любом интерфейсе без переходников. Исходники не меняются.",
                              "Re-lays a LoRA file onto the other architecture by the official expansion layout: "
                              "2B blocks go to their places in 2.9B. The result is a normal LoRA usable in any UI "
                              "without adapters. Source files are not changed."), elem_classes="small-note")
                with gr.Row():
                    v_files = gr.Dropdown(label=t("LoRA", "LoRA"), choices=scan_files(SET["lora_dir"], "lora"),
                                          multiselect=True, value=[], allow_custom_value=True, filterable=True, scale=3)
                    mlds.append(v_files)
                    v_folder = gr.Textbox(label=t("…или вся папка", "…or a whole folder"), scale=2)
                with gr.Row():
                    v_to = gr.Radio([(t("авто (в другую)", "auto (to the other one)"), "auto"), ("2B → 2.9B", "40"),
                                     ("2.9B → 2B", "28")], value="auto", label=t("Направление", "Direction"))
                    v_mins = gr.Slider(0, 1, value=0, step=0.05,
                                       label=t("2B → 2.9B: копия во вставленные блоки", "2B → 2.9B: copy into inserted blocks"),
                                       info=t("0 = вставленные блоки пустые (как при вшивании по умолчанию). 0.2–0.5 — "
                                              "ослабленная копия соседнего блока", "0 = inserted blocks stay empty (like the bake default). "
                                              "0.2–0.5 — a weakened copy of the neighbour block"))
                v_out = gr.Textbox(label=t("Папка для результатов (пусто = рядом с исходником, суффикс _2.9B / _2B)",
                                           "Output folder (empty = next to the source, suffix _2.9B / _2B)"))
                v_go = gr.Button(t("▶ Сконвертировать", "▶ Convert"), variant="primary")

            # ---------------- analysis
            with gr.Tab(t("Анализ", "Analysis")):
                gr.Markdown(t("Анализаторы только читают файлы. Отчёт открывается ниже и сохраняется в папку `reports/` "
                              "(HTML с графиками + текст + JSON). На видеокарте считается в разы быстрее.",
                              "Analyzers only read files. The report opens below and is saved to the `reports/` folder "
                              "(HTML with charts + text + JSON). Much faster on a GPU."), elem_classes="small-note")
                with gr.Row():
                    a_dev = gr.Radio(DEVICES, value=SET.get("device", "auto"), label=t("Устройство", "Device"))
                    a_exact = gr.Checkbox(False, label=t("Точно, без профилей (медленно)", "Exact, no profiles (slow)"),
                                          info=t("По умолчанию модели сканируются один раз, их профиль сохраняется в "
                                                 "profiles/, и следующие анализы идут за секунды (погрешность 1–2%).",
                                                 "By default each model is scanned once and its profile is saved to "
                                                 "profiles/; later analyses take seconds (within 1–2%)."))
                with gr.Accordion(t("Профили моделей", "Model profiles"), open=False):
                    gr.Markdown(t("Профиль — небольшой «слепок» модели (около 7 МБ для 2.9B). Строится при первом анализе "
                                  "модели, а дальше анализ читает только его. Можно построить заранее для всей папки, "
                                  "например на ночь: потом любой анализ этих моделей займёт секунды. Изменённый файл "
                                  "получит новый профиль сам, переименованный сохранит старый.",
                                  "A profile is a small 'snapshot' of a model (about 7 MB for 2.9B). It is built on the "
                                  "model's first analysis, and later analyses read only the profile. You can build them in "
                                  "advance for a whole folder, e.g. overnight: then any analysis of those models takes "
                                  "seconds. A changed file gets a new profile by itself; a renamed one keeps its profile."),
                                elem_classes="small-note")
                    a_pinfo = gr.Markdown(profiles_summary())
                    with gr.Row():
                        a_pdir = gr.Textbox(label=t("Папка с моделями", "Models folder"), value=SET["models_dir"], scale=3,
                                            info=t("по умолчанию — папка моделей сверху", "defaults to the models folder above"))
                        a_preb = gr.Checkbox(False, label=t("Пересобрать даже готовые", "Rebuild existing ones too"), scale=1)
                    with gr.Row():
                        a_pbuild = gr.Button(t("▶ Построить профили для папки", "▶ Build profiles for the folder"),
                                             variant="primary")
                        a_pref = gr.Button(t("Обновить сводку", "Refresh summary"), size="sm")
                    with gr.Row():
                        a_pok = gr.Checkbox(False, label=t("Да, удалить все профили", "Yes, delete all profiles"))
                        a_pclr = gr.Button(t("Очистить профили", "Clear profiles"), variant="stop", size="sm")
                with gr.Tabs():
                    with gr.Tab(t("Родитель", "Parent")):
                        gr.Markdown(t("От какой модели сделана цель. Пусто в кандидатах = все модели той же архитектуры в папке.",
                                      "Which model the target was made from. Empty candidates = all models of the same "
                                      "architecture in the folder."), elem_classes="small-note")
                        pT = model_dd(t("Цель", "Target"))
                        pC = models_multi(t("Кандидаты", "Candidates"))
                        pick_buttons(pC)
                        pB = model_dd(t("Официальная база (необязательно, для масштаба)", "Official base (optional, for scale)"))
                        guess_btn(pC, pB)
                        p_go = gr.Button(t("▶ Найти родителя", "▶ Find the parent"), variant="primary")
                    with gr.Tab(t("Семья", "Family")):
                        gr.Markdown(t("Кто от кого: расстояния, ближайшие родственники, одинаковые тензоры.",
                                      "Who comes from whom: distances, nearest relatives, identical tensors."),
                                    elem_classes="small-note")
                        fM = models_multi(t("Модели", "Models"))
                        pick_buttons(fM)
                        fR = model_dd(t("Эталон (обычно официальная база)", "Reference (usually the official base)"))
                        guess_btn(fM, fR)
                        f_go = gr.Button(t("▶ Построить древо", "▶ Build the tree"), variant="primary")
                    with gr.Tab(t("Геном / состав", "Genome / composition")):
                        gr.Markdown(t("Общие «гены» семейства и из чего собран каждый мердж (с весами и поблочно).",
                                      "Shared 'genes' of a family and what each merge is assembled from (with weights, per block)."),
                                    elem_classes="small-note")
                        gM = models_multi(t("Модели", "Models"))
                        pick_buttons(gM)
                        gB = model_dd(t("База", "Base"))
                        guess_btn(gM, gB)
                        g_go = gr.Button(t("▶ Расшифровать", "▶ Decode"), variant="primary")
                    with gr.Tab(t("Блоки", "Blocks")):
                        gr.Markdown(t("Где модели отличаются от эталона: по блокам (вставленные блоки 2.9B отмечены), "
                                      "по типам слоёв и частям вне блоков, плюс проверка устройства файлов.",
                                      "Where models differ from a reference: per block (inserted 2.9B blocks marked), "
                                      "per layer type and non-block part, plus a file structure check."),
                                    elem_classes="small-note")
                        kM = models_multi(t("Модели", "Models"))
                        pick_buttons(kM)
                        kR = model_dd(t("Эталон", "Reference"))
                        guess_btn(kM, kR)
                        kS = model_dd(t("Для 2.9B: база 2B (необязательно)", "For 2.9B: a 2B base (optional)"),
                                      t("покажет, трогали ли старые 28 блоков", "shows whether the original 28 blocks were touched"))
                        k_go = gr.Button(t("▶ Разобрать", "▶ Analyze"), variant="primary")
                    with gr.Tab(t("Рецепт / метаданные", "Recipe / metadata")):
                        gr.Markdown(t("Как сделана модель: шаги AnimaMerger, рецепты других мерджеров, параметры обучения LoRA. "
                                      "Читаются только заголовки — мгновенно.",
                                      "How a model was made: AnimaMerger steps, other mergers' recipes, LoRA training "
                                      "parameters. Only headers are read — instant."), elem_classes="small-note")
                        rM = models_multi(t("Файлы", "Files"))
                        with gr.Row():
                            r_all = gr.Button(t("Все модели в папке", "All models in the folder"), size="sm")
                            r_sha = gr.Checkbox(label=t("SHA256 (медленно, для сверки с Civitai)", "SHA256 (slow, to match Civitai)"))
                        r_all.click(lambda d: [p for _, p in scan_files(d)], models_dir, rM)
                        r_go = gr.Button(t("▶ Показать", "▶ Show"), variant="primary")
                    with gr.Tab(t("Анализ LoRA", "LoRA analysis")):
                        gr.Markdown(t("Одна LoRA — сила, где работает, датасет по тегам. Несколько чекпоинтов одного "
                                      "обучения — график по шагам/эпохам: где LoRA «выучилась» и где дальше растёт только сила "
                                      "(риск пережога). Номера шагов берутся из имён (-step500, -000010) или метаданных.",
                                      "One LoRA — strength, where it works, dataset by tags. Several checkpoints of one training — "
                                      "a curve over steps/epochs: where the LoRA 'learned' and where only the strength keeps "
                                      "growing (over-baking risk). Step numbers come from file names (-step500, -000010) or metadata."),
                                    elem_classes="small-note")
                        with gr.Row():
                            lF = gr.Dropdown(label=t("LoRA", "LoRA"), choices=scan_files(SET["lora_dir"], "lora"),
                                             multiselect=True, value=[], allow_custom_value=True, filterable=True, scale=3)
                            mlds.append(lF)
                            lD = gr.Textbox(label=t("…или папка с чекпоинтами", "…or a folder with checkpoints"), scale=2)
                        with gr.Row():
                            lS = gr.Textbox(label=t("Номера шагов вручную (необязательно)", "Step numbers by hand (optional)"),
                                            placeholder="500, 1000, 1500",
                                            info=t("по алфавиту имён файлов", "in file-name order"))
                            lM = model_dd(t("Модель для оценки силы (необязательно)", "Model to measure strength against (optional)"))
                        l_go = gr.Button(t("▶ Анализировать", "▶ Analyze"), variant="primary")
                rep_file = gr.File(label=t("Файл отчёта (открой в браузере)", "Report file (open in a browser)"), interactive=False)
                rep_html = gr.HTML()

            # ---------------- recipe
            with gr.Tab(t("Рецепт", "Recipe")):
                gr.Markdown(RECIPE_HELP, elem_classes="small-note")
                r_text = gr.Textbox(label=t("Рецепт", "Recipe"), lines=10, elem_id="recipe",
                                    placeholder="Flat = merge a=... b=... c=... m=0.5 deep=\"0, L16-L27:mlp:0.5\"")
                with gr.Row():
                    r_clean = gr.Checkbox(label=t("Удалить промежуточные файлы после успешного завершения",
                                                  "Delete intermediate files after success"),
                                          info=t("Удаляются только шаги, на которые ссылаются через $; конечные результаты остаются",
                                                 "Only steps referenced via $ are deleted; final results stay"))
                    r_check = gr.Button(t("Проверить", "Check"))
                    r_run = gr.Button(t("▶ Выполнить рецепт", "▶ Run the recipe"), variant="primary")

            # ---------------- cross
            with gr.Tab(t("Перенос 2B ↔ 2.9B", "Port 2B ↔ 2.9B")):
                gr.Markdown(t("Переносит дельту файнтюна на модель другой архитектуры по карте блоков. "
                              "Направление определяется автоматически.",
                              "Ports a fine-tune's delta onto a model of the other architecture using a block map. "
                              "The direction is detected automatically."), elem_classes="small-note")
                with gr.Row():
                    cT = model_dd(t("Target — модель другой архитектуры", "Target — a model of the other architecture"))
                    cF = model_dd(t("Файнтюн, который переносим", "The fine-tune to port"))
                    cFB = model_dd(t("База, от которой обучался файнтюн", "The base the fine-tune was trained from"))
                with gr.Group():
                    c_src = gr.Radio(["spec", "auto", "file"], value="spec", label=t("Источник карты блоков", "Block map source"),
                                     info=t("spec — по схеме (official — опубликованная схема расширения); auto — по весам; file — готовая карта",
                                            "spec — by a layout (official = the published expansion); auto — from the weights; file — a saved map"))
                    with gr.Row():
                        c_ref = model_dd(t("auto: эталонная база той же архитектуры, что target", "auto: a reference base of the target's architecture"),
                                         t("нужна, если target — файнтюн, а не база", "needed if the target is a fine-tune, not a base"))
                    with gr.Row():
                        c_preset = gr.Dropdown(PRESETS_UI, value="official", label=t("spec: схема", "spec: layout"))
                        c_ins = gr.Textbox(label=t("spec: свои позиции вставленных слоёв (0-based)", "spec: custom inserted positions (0-based)"),
                                           placeholder=t("пусто = по выбранной схеме", "empty = by the chosen layout"))
                        c_copy = gr.Radio(["prev", "next"], value="prev", label=t("spec (не official): копия какого соседа",
                                                                                  "spec (not official): copy of which neighbour"))
                    with gr.Row():
                        c_map = gr.Dropdown(label=t("file: карта", "file: map"), choices=scan_maps(), value=None, allow_custom_value=True)
                        c_save_map = gr.Checkbox(label=t("Сохранить использованную карту в maps/", "Save the used map to maps/"), value=True)
                with gr.Row():
                    c_mins = gr.Slider(0, 1, value=0.0, step=0.05, label=t("Вверх: доля дельты для вставленных слоёв",
                                                                           "Up: delta share for inserted layers"),
                                       info=t("0 = не трогать (безопасно), 0.2–0.5 = осторожно", "0 = leave alone (safe), 0.2–0.5 = careful"))
                    c_isrc = gr.Radio(["copy", "prev", "next", "avg"], value="copy",
                                      label=t("Вверх: чья дельта для вставленных слоёв", "Up: whose delta the inserted layers get"))
                    c_fold = gr.Slider(0, 1, value=0.0, step=0.05,
                                       label=t("Вниз: влить дельты вставленных слоёв в оригиналы", "Down: fold inserted-layer deltas into the originals"))
                with gr.Row():
                    c_ms = gr.Textbox(label=t("M (можно несколько: 0.3, 0.5, 0.7)", "M (several allowed: 0.3, 0.5, 0.7)"), value="0.8")
                    c_name = gr.Textbox(label=t("Имя выходного файла (пусто = автоматически)", "Output file name (empty = automatic)"))
                    c_dtype = gr.Dropdown(DTYPES_UI, value="same", label=t("Тип сохранения", "Save as"))
                with gr.Accordion(t("Дополнительные множители", "More multipliers"), open=False):
                    with gr.Row():
                        c_adaln = gr.Number(label=t("M для adaLN (пусто = как M, 0 = оставить от target)",
                                                    "M for adaLN (empty = as M, 0 = keep target)"), value=None)
                        c_nonb = gr.Number(label=t("M для не-блочных весов (пусто = как M)", "M for non-block weights (empty = as M)"),
                                           value=None)
                    c_blk = gr.Textbox(label=t("Поблочные коэффициенты (× M), индексы target", "Per-block factors (× M), target indexes"),
                                       placeholder="0-9:0.5,30-39:1.2")
                    c_deep = gr.Textbox(label=t("Точные правила по блокам и слоям (индексы target)", "Exact rules per block and layer (target indexes)"),
                                        lines=2, placeholder="0, L16-L27:mlp.layer1 mlp.layer2:{M}", info=DEEP_INFO)
                with gr.Row():
                    c_dry = gr.Checkbox(label=t("Пробный прогон (не сохранять)", "Dry run (don't save)"))
                    c_go = gr.Button(t("▶ Перенести", "▶ Port"), variant="primary")

            # ---------------- map
            with gr.Tab(t("Карта блоков", "Block map")):
                gr.Markdown(t("Построить карту 2B ↔ 2.9B и сохранить в `maps/`. compare считает карту по весам и сверяет её со схемой.",
                              "Build a 2B ↔ 2.9B map and save it to `maps/`. compare computes the map from the weights "
                              "and checks it against the layout."), elem_classes="small-note")
                with gr.Row():
                    pS = model_dd(t("База 2B (28 блоков)", "2B base (28 blocks)"))
                    pL = model_dd(t("База 2.9B (40 блоков)", "2.9B base (40 blocks)"))
                with gr.Row():
                    p_src = gr.Radio(["auto", "spec", "compare"], value="compare", label=t("Источник", "Source"))
                    p_keep = gr.Radio(["auto", "spec"], value="spec", label=t("compare: какую сохранить", "compare: which to keep"))
                with gr.Row():
                    p_preset = gr.Dropdown(PRESETS_UI, value="official", label=t("spec: схема", "spec: layout"))
                    p_ins = gr.Textbox(label=t("spec: свои позиции (0-based)", "spec: custom positions (0-based)"),
                                       placeholder=t("пусто = по выбранной схеме", "empty = by the chosen layout"))
                p_name = gr.Textbox(label=t("Имя файла карты", "Map file name"), value="anima_map")
                p_mgo = gr.Button(t("▶ Построить карту", "▶ Build the map"), variant="primary")

            # ---------------- info
            with gr.Tab(t("Инфо о модели", "Model info")):
                iM = model_dd(t("Модель", "Model"))
                i_go = gr.Button(t("▶ Показать структуру", "▶ Show structure"))

            # ---------------- expert
            with gr.Tab(t("Эксперт", "Expert")):
                e_sample = gr.Number(label=t("auto-карта: элементов тензора для оценки схожести",
                                             "auto map: tensor elements to sample"), value=1_000_000, precision=0)
                e_out = gr.Textbox(label=t("Шаблоны выходных проекций (через пробел, пусто = по умолчанию)",
                                           "Output projection patterns (space-separated, empty = default)"),
                                   placeholder="output_proj out_proj to_out proj_out o_proj mlp.layer2 fc2 linear2")
                e_adaln = gr.Textbox(label=t("Шаблоны слоёв модуляции (пусто = по умолчанию)", "Modulation layer patterns (empty = default)"),
                                     placeholder="adaln adaLN modulation")
                gr.Markdown(t("Карты: ", "Maps: ") + f"`{MAP_DIR}`  \n" + t("Отчёты: ", "Reports: ") + f"`{ROOT / 'reports'}`  \n"
                            + t("Настройки: ", "Settings: ") + f"`{SETTINGS_FILE}`", elem_classes="small-note")

        with gr.Row():
            stop_btn = gr.Button(t("■ Стоп", "■ Stop"), variant="stop", size="sm", scale=0)
            prog = gr.HTML()
        log = gr.Textbox(label=t("Лог", "Log"), lines=22, max_lines=40, autoscroll=True, elem_id="log",
                         show_copy_button=True)

        # --- events
        def refresh_all(mdir, ldir, odir):
            save_settings(mdir, ldir, odir)
            ch, lch, lonly = scan_files(mdir), scan_files(ldir), scan_files(ldir, "lora")
            return ([gr.update(choices=ch) for _ in dds] + [gr.update(choices=ch) for _ in mdds]
                    + [gr.update(choices=lch) for _ in ldds] + [gr.update(choices=lonly) for _ in mlds]
                    + [gr.update(choices=scan_maps())])

        folders = [models_dir, lora_dir, out_dir_tb]
        targets = dds + mdds + ldds + mlds + [c_map]
        refresh.click(refresh_all, folders, targets)
        for tb in folders:
            tb.submit(refresh_all, folders, targets)
            tb.blur(lambda a, b, c: save_settings(a, b, c), folders, None)

        lang.change(switch_lang, lang, None, js="(v) => { setTimeout(() => location.reload(), 6000); return v; }")

        def remember_dev(d):
            SET["device"] = d
            write_settings()
        a_dev.change(remember_dev, a_dev, None)

        LP = [log, prog]
        LR = [log, prog, rep_html, rep_file]
        m_go.click(do_merge, [mA, mB, mC, m_ms, m_adaln, m_nonb, m_blk, m_name, m_dtype, m_dry,
                              e_out, e_adaln, m_deep, m_mode, m_dens, m_seed], LP)
        b_add.click(add_lora_row, [b_rows, b_pick, b_str], b_rows)
        b_go.click(do_bake, [bM, b_rows, b_deep, b_cap, b_mins, b_name, b_dtype, b_dry, b_grid], LP)
        x_go.click(do_extract, [xT, xB, x_rank, x_ra, x_deep, x_nonb, x_clamp, x_dev, x_name, x_dtype, x_dry], LP
                   ).then(lambda d: gr.update(choices=scan_files(d)), lora_dir, b_pick)
        j_add.click(add_lora_row, [j_rows, j_pick, j_str], j_rows)
        j_go.click(do_lora_merge, [j_rows, j_deep, j_mode, j_rank, j_dev, j_name, j_dtype, j_dry], LP
                   ).then(lambda d: gr.update(choices=scan_files(d)), lora_dir, b_pick)
        v_go.click(do_convert, [v_files, v_folder, v_to, v_mins, v_out], LP
                   ).then(lambda d: gr.update(choices=scan_files(d)), lora_dir, b_pick)
        models_dir.change(lambda d: d, models_dir, a_pdir)
        a_pbuild.click(do_profiles_build, [a_pdir, a_preb], LP).then(profiles_summary, None, a_pinfo)
        a_pref.click(profiles_summary, None, a_pinfo)
        a_pclr.click(do_profiles_clear, a_pok, [a_pinfo, a_pok])
        p_go.click(do_parent, [pT, pC, pB, a_dev, a_exact], LR).then(profiles_summary, None, a_pinfo)
        f_go.click(do_family, [fM, fR, a_dev, a_exact], LR).then(profiles_summary, None, a_pinfo)
        g_go.click(do_genome, [gM, gB, a_dev, a_exact], LR).then(profiles_summary, None, a_pinfo)
        k_go.click(do_blocks, [kM, kR, kS, a_dev, a_exact], LR).then(profiles_summary, None, a_pinfo)
        r_go.click(do_meta, [rM, r_sha], LR)
        l_go.click(do_lora_info, [lF, lD, lS, lM, a_dev], LR)
        r_check.click(check_recipe, r_text, log)
        r_run.click(do_recipe, [r_text, r_clean], LP)
        c_go.click(do_cross, [cT, cF, cFB, c_src, c_ref, c_ins, c_preset, c_copy, c_map, c_save_map,
                              c_mins, c_isrc, c_fold, c_ms, c_adaln, c_nonb, c_blk, c_name, c_dtype,
                              c_dry, e_sample, e_out, e_adaln, c_deep], LP
                   ).then(lambda: gr.update(choices=scan_maps()), None, c_map)
        p_mgo.click(do_map, [pS, pL, p_src, p_ins, p_preset, p_keep, p_name, e_sample, e_out, e_adaln], LP
                    ).then(lambda: gr.update(choices=scan_maps()), None, c_map)
        i_go.click(do_info, [iM, e_out, e_adaln], LP)
        stop_btn.click(stop, None, None, queue=False)
    return ui


def kill_children_with_app():
    """Windows: put the app into a job object, so merge/analysis processes die when the app is closed
    (closing the console window otherwise leaves them running and holding tens of GB of RAM)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = EXT()
        info.Basic.LimitFlags = 0x2000 | 0x800  # KILL_ON_JOB_CLOSE | BREAKAWAY_OK (for the language restart)
        if k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
        globals()["_JOB"] = job  # keep the handle open for the app's lifetime
    except Exception:
        pass


if __name__ == "__main__":
    kill_children_with_app()
    port = int(os.environ.get("ANIMA_MERGER_PORT", "7887"))
    build().queue().launch(server_name="127.0.0.1", server_port=port,
                           **({"css": CSS, "theme": gr.themes.Soft()} if STYLE_AT_LAUNCH else {}),
                           inbrowser=os.environ.get("ANIMA_NO_BROWSER") != "1")
