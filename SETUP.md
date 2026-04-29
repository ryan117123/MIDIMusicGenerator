# Setup instructions

Follow these steps **in order**. Use **Python 3.10+** (3.11 recommended). GPU is optional for notebooks but speeds training; the Gradio demo runs on CPU or GPU.

---

## 1. Get the repository

```bash
git clone <your-fork-or-repo-url>.git
cd RScs372_finalproject
```

(If you downloaded a ZIP, unzip it and `cd` into the project folder instead.)

---

## 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

You should see `(.venv)` in your shell prompt when the environment is active.

---

## 3. Install Python dependencies

From the **project root** (the directory that contains `requirements.txt`):

```bash
pip install -r requirements.txt
```

**Verify**

```bash
python -c "import torch, gradio, pretty_midi; print('OK', torch.__version__)"
```

If this prints `OK` and a PyTorch version, core packages are installed.

---

## 4. Prepare data (MIDI corpus)

1. Obtain **piano MIDI** files and any metadata your notebooks expect (see `notebooks/01_data_audit_preprocess.ipynb` for directory layout).
2. Place raw inputs under **`data/raw/`** (create the folder if needed). Large downloads are not committed to git.
3. Run notebook **01** onward to build processed tables, caches, and checkpoints under **`data/project_outputs/`**.

---

## 5. Run the analysis pipeline (notebooks)

1. Start Jupyter: `jupyter lab` or open notebooks in **VS Code / Cursor**.
2. Run notebooks in the order listed in **`notebooks/README.md`** (01 → … → 07).
3. Prefer **GPU** for training cells when available.

---

## 6. Run the Gradio demo (local)

After you have a **checkpoint** `.pt` and a **sequence cache** JSON (from your training pipeline):

```bash
python scripts/gradio_demo.py \
  --checkpoint data/project_outputs/cache/07_transformer_<your_run>.pt \
  --cache data/project_outputs/cache/sequence_cache_256seq.json
```

- Run from the **repository root** so relative paths resolve.
- Use `python scripts/gradio_demo.py --help` for `--seq-len`, temperature, and time-shift decoding flags.
- **Audio:** rendering uses `pretty_midi` (FluidSynth or SciPy synthesis depending on your install). If audio fails, install **FluidSynth** and **pyfluidsynth**, or ensure **SciPy** is available (see `requirements.txt`).

---

## 7. Google Colab (optional)

1. Upload the `notebooks/` folder or open notebooks from GitHub.
2. Runtime → **Change runtime type** → **GPU** (recommended for training).
3. In the first code cell:  
   `!pip install -r requirements.txt`  
   (Upload `requirements.txt` or paste package names from it.)
4. Mount **Google Drive** if your data and outputs live on Drive; update paths in the notebooks to match.
5. Download your MIDI corpus and set the root path in notebook **01** config.

---

## Reproducibility

- Training notebooks set **random seeds** where noted.
- Save **checkpoints** (`.pt`) and **CSV** exports under `data/project_outputs/cache/` and `data/project_outputs/tables/` (or your Drive mirror).
- Copy important figures into `data/project_outputs/figures/` or `reports/figures/` for submission.
