# MIDI continuation (CS372 final project)

## What it Does

This project builds a **neural next-token model** for **piano MIDI continuation**: performances are converted into a discrete sequence of tokens (note on/off events and quantized **`TIME_SHIFT_*`** steps). A **Transformer** reads a fixed-length context window and predicts the distribution over the **next** token. You can **train** and **evaluate** the pipeline in Jupyter notebooks (exported metrics as CSV), then load a saved checkpoint into a **Gradio** web app that uploads a MIDI file, slices context from a chosen start time, and plays **unigram baseline** vs **Transformer** continuations as audio for qualitative comparison.

---

## Quick Start

1. **Install:** Follow **`SETUP.md`** (create a venv, then `pip install -r requirements.txt` from the repo root).
2. **Data:** Put your piano MIDI corpus under **`data/raw/`** as described in **`notebooks/01_data_audit_preprocess.ipynb`** (see also **`ATTRIBUTION.md`** for the public dataset source).
3. **Train / evaluate:** Open Jupyter, then run notebooks **01 → 07** in order (see **`notebooks/README.md`**). Outputs land under **`data/project_outputs/tables/`**, **`figures/`**, and **`cache/`**.
4. **Demo:** With a trained **`.pt`** checkpoint and matching **sequence cache JSON**:
   ```bash
   python scripts/gradio_demo.py \
     --checkpoint data/project_outputs/cache/07_transformer_<your_run>.pt \
     --cache data/project_outputs/cache/sequence_cache_256seq.json
   ```
   Use **`--seq-len 256`** if you trained with 256-token windows. Run **`python scripts/gradio_demo.py --help`** for temperature and time-shift decoding flags.

---

## Repository layout

| Path | Purpose |
|------|--------|
| `notebooks/` | Pipelines 01–07: data → features → baselines → custom model → hparams → comparison → final Transformer. |
| `scripts/gradio_demo.py` | Gradio UI: MIDI upload, start time, baseline vs Transformer audio. |
| `data/raw/` | Raw MIDI inputs (not committed; `.gitkeep` only). |
| `data/project_outputs/cache/` | Caches, experiment JSON, **`.pt`** checkpoints. |
| `data/project_outputs/tables/` | Exported metrics and training histories (CSV). |
| `data/project_outputs/figures/` | Exported plots. |
| `SETUP.md` | Step-by-step installation. |
| `ATTRIBUTION.md` | Dataset + AI disclosure. |
| `requirements.txt` | Python dependencies. |

---

## Evaluation

### Quantitative (saved runs)

Results are exported as CSV under **`data/project_outputs/tables/`**. Your numbers will depend on the checkpoint and split; **examples** from committed exports:

| Source file | Takeaway (example values) |
|-------------|---------------------------|
| **`03_baseline_results.csv`** | On **quantized_time**, unigram **~12.6%** next-token accuracy vs bigram Markov **~13.4%** (teacher-forced one-step prediction on the test split). |
| **`06_model_comparison.csv`** | **Transformer** (`transformer_custom`) **~28.5%** accuracy, CE **~2.74**, top-5 **~64.5%** vs **GRU** **~28.0%** / CE **~2.74** on `quantized_time`; both far above n-gram baselines on the same setting. |
| **`07_training_metrics_*.csv`** | Final Transformer run (example slug `...031737...`): test **accuracy ~27.4%**, **cross-entropy ~2.87**, **perplexity ~17.6**, **top-5 ~63.2%**; **best val loss ~2.93** at epoch **10** (see that row for `n_train_windows` / `n_test_windows`). |

Re-run notebooks to regenerate CSVs if your splits or code changed; cite the exact file name you submit.

### Qualitative

- **Token dumps:** `data/project_outputs/cache/04_generated_tokens_quantized.txt` (and related files) for raw generation inspection.
- **Listening:** **`scripts/gradio_demo.py`** — compare baseline vs model continuations on the same MIDI prefix; note rhythmic / harmonic artifacts in your report or walkthrough.

### Limitations (one line)

Metrics are **single-step** accuracy under **teacher forcing**; long **autoregressive** rollouts in the demo can behave differently—report both tables and audio.

---

## Video Links

**Before you submit**, replace the placeholders below with **direct links** to your hosted recordings (for example unlisted YouTube or a shared Drive file with view access).

| Video | Link |
|-------|------|
| **Demo** | https://drive.google.com/file/d/1tAPKY3B_fK9GA4jxVfVdw-oHH0y6ECB6/view?usp=drive_link|
| **Technical walkthrough** | https://drive.google.com/file/d/1I6kuhshuP8MU68YLEgSMsPkboixWhg3s/view?usp=sharing|

Optional talking-point checklist: **`VIDEO_RECORDING_SCRIPTS.md`**.

---

## License / data

Third-party MIDI corpora have their own licenses; cite your data source in any report and in **`ATTRIBUTION.md`**.
