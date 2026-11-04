# Anima Merger 1.1

A local tool for **Anima** image models (2B, 28 blocks, and 2.9B, 40 blocks): merging, baking and extracting LoRAs, porting fine-tunes between 2B and 2.9B, converting LoRA files, and **analyzing** models and LoRAs with visual HTML reports.

Everything runs on your machine. Source files are only read; results are always written as new files.

🇷🇺 [Русская версия](README_RU.md)

## Quick start

**Windows**
1. Unpack the folder anywhere (preferably a path without non-Latin characters).
2. Run `install.bat`. It creates a `venv` and installs torch (CPU build, ~200 MB), gradio and safetensors. Python 3.10–3.12 is required.
3. Optional, recommended for NVIDIA cards: run `install_cuda.bat` to switch to the GPU build of torch. Analysis and LoRA extraction become many times faster.
4. Run `run.bat`. The UI opens at `http://127.0.0.1:7887`.

**Linux:** `./install.sh`, then `./run.sh`.

Set your models, LoRA and output folders at the top and press **Refresh lists**. Folders are remembered in `settings.json`. The language switch (English / Русский) is at the top right; the app restarts itself to apply it.

## What it does

| Tab | What for |
|---|---|
| **Merge / rebase** | `A + (B − C) × M` for models of the same architecture. Modes: `add`, `topk` (only the strongest changes), `dare`, `ortho` (without the part parallel to A's weights), `ortho_delta` (without what A already took from C: no double dose when B and A share a parent; `--proj` sets how much to remove), `normdir`. Several M values at once → one file per value. |
| **Bake LoRA** | Bake one or more LoRA / LoHa / LoKr into a model. kohya, diffusion-pipe and ComfyUI formats. Per-block / per-layer multipliers, a limiter against over-baking, a strength grid. A 2B LoRA on a 2.9B model is mapped automatically. |
| **Extract LoRA** | A LoRA from the difference of two models (SVD) with an accuracy estimate. |
| **Mix LoRAs** | Mix several LoRAs into one file without a model: `concat` (exact, rank = sum of ranks) or `svd` (compressed to a chosen rank, accuracy shown). Per-block rules, e.g. composition from one LoRA and line/texture from another. |
| **LoRA converter 2B ↔ 2.9B** | Re-lays LoRA files onto the other architecture by the official expansion layout. The result is a normal LoRA usable in any UI. Batch: a whole folder at once, subfolders included (with an output folder, the subfolder layout is kept; earlier results next to their sources are skipped). |
| **Analysis** | Read-only analyzers with HTML reports (below). |
| **Recipe** | A chain of steps in text (`Flat = merge ...`, then `Final = bake model=$Flat ...`), optional cleanup of intermediate files. |
| **Port 2B ↔ 2.9B** | Port a fine-tune's delta to the other architecture with a block map (`official` layout, computed from the weights, or a saved map). |
| **Block map** | Build and save a 2B ↔ 2.9B block map; `compare` checks the computed map against the layout. |
| **Model info** | Structure of a checkpoint: prefix, blocks, layer names. |

### Analysis

| Analyzer | Answers |
|---|---|
| **Parent** | Which model a fine-tune or merge was made from. Ranking with a confidence verdict, per-block "stripes" for block-wise merges, where the training went. |
| **Family** | A family tree: distance matrix, nearest relatives, bit-identical tensors and fully identical blocks (direct parent / child). |
| **Genome / composition** | Shared "genes" of a family (common style vs. branch splits) and **what each merge is assembled from**, with weights — globally and per block (detects block-weighted merges). |
| **Blocks** | Where models differ from a reference: per block (the 12 inserted 2.9B blocks are marked), per layer type, non-block parts; for 2.9B, whether the original 28 blocks were touched; a file-structure check. |
| **Recipe / metadata** | AnimaMerger steps stored in the file, other mergers' recipes, LoRA training parameters, optional SHA256. Instant: only headers are read. |
| **LoRA analysis** | One LoRA: strength (optionally relative to a model), where it works (blocks, layer types), effective rank, training parameters and **dataset tags** (tags on every image, "junk" tags like signatures and watermarks). Several checkpoints of one training: a curve over steps/epochs that marks where the LoRA "learned" and where only its strength keeps growing (over-baking risk). Step numbers come from file names (`-step500`, `-000010`, `_e10`), metadata, or can be typed in. |

Reports are saved to `reports/` as `.html` (charts; hover for values), `.txt` and `.json`, and are also shown right in the UI. On a GPU they're computed there, otherwise on the CPU.

**Profiles (fast re-analysis).** Parent, Family, Genome and Blocks work from model profiles by default. The first time a model is analyzed it is read once and a small profile (about 7 MB for a 2.9B model) is saved to `profiles/`: the file's fingerprint, its structure, an exact hash of every tensor and a fixed sample of 4096 weights from each tensor. Every later analysis reads only the profiles, so adding one new model to a family of ten means scanning that one model, not all eleven. Distances, shares and merge weights from profiles match the exact run within about 1–2%; identical tensors are found exactly. A renamed or moved file keeps its profile; a changed file gets a new one. For an exact run tick **Exact, no profiles** (CLI: `--exact`). To build profiles in advance (e.g. overnight), open **Analysis → Model profiles** and press **Build profiles for the folder**, or run `python anima_merger.py profile --models <folder>`. The same block shows how many profiles there are and how much space they take, and can clear them.

> LoRA analysis works on the weights. Its under/over-training verdicts are heuristics; images on a fixed seed have the final word.

## Rules per block and layer (`deep`)

As in Forge Neo recipes: `0.5, L16-L27:mlp.layer1 mlp.layer2:0.2, L00-L27:adaln:0`
- a bare number first is the default multiplier;
- `Lx-Ly:patterns:value` — blocks, layer-name substrings separated by spaces, value; the last matching rule wins;
- `adapter:*:0` / `adapter:self_attn:0.5` — rules for the LLM adapter (`llm_adapter`);
- `^cross_attn` — match only from the start of the layer name. Plain `cross_attn` also matches `adaln_modulation_cross_attn`; the tool warns when a rule catches adaLN layers like that.

## Command line

Every UI action is a command: `python anima_merger.py <command> --help`

`merge`, `bake`, `extract`, `lora-merge`, `lora-convert`, `cross`, `map`, `info`, `parent`, `family`, `genome`, `blocks`, `meta`, `lora-info`, `profile`.

```
python anima_merger.py bake --model MyModel.safetensors --lora style.safetensors:0.7 --out MyModel+style.safetensors
python anima_merger.py lora-convert --lora my_loras_folder --to 40
python anima_merger.py lora-info --loras training_output_folder --model anima_base.safetensors
python anima_merger.py genome --models models_folder --base anima_base.safetensors
```

Set `ANIMA_LANG=en` or `ru` for the language of the CLI messages, `ANIMA_MERGER_PORT` to change the UI port.

## Notes

- Saving shows progress and writes `name.safetensors.part` first; an interrupted save never leaves a half-written model under the real name.
- Merging/baking/porting results store how they were made in the file header (`anima_merger` field) — see **Recipe / metadata**.
- Porting between 2B and 2.9B is approximate: after further training, 2.9B blocks are no longer equal to 2B blocks. Always check results on a fixed seed with a grid of M values.

## Dependencies and licenses

[PyTorch](https://pytorch.org) (BSD-3-Clause), [safetensors](https://github.com/huggingface/safetensors) (Apache-2.0), [Gradio](https://gradio.app) (Apache-2.0), [NumPy](https://numpy.org) (BSD-3-Clause). Anima itself is a separate model with its own license; this tool contains no model weights.

## License

AnimaMerger is free software under the [GNU GPL v3](LICENSE): you may use, change and share it, and changed versions you share must stay open under the same license.
