# Notebook order

Run these **in order** so caches and tables line up. Paths in later notebooks often assume outputs from earlier ones (under `data/project_outputs/` or paths you set in Colab).

| Order | Notebook | Role |
|------:|----------|------|
| 1 | `01_data_audit_preprocess.ipynb` | MIDI data audit, splits, preprocessing / quantization tables |
| 2 | `02_feature_engineering_dimred.ipynb` | Sequence features, representation, PCA / selection summaries |
| 3 | `03_baseline_models.ipynb` | Baseline models and `03_baseline_results.csv` |
| 4 | `04_custom_model_training.ipynb` | Custom model training experiments and logs |
| 5 | `05_hparam_and_regularization.ipynb` | Hyperparameter and regularization search |
| 6 | `06_model_comparison_ablation.ipynb` | Comparisons and ablation tables |
| 7 | `07_final_transformer_training.ipynb` | Final quantized-time Transformer training + optional MIDI export cells |

## Evidence for rubric

Each notebook should end (or include a short section) titled **“Evidence for Rubric”** listing:

- Exported **CSV** paths under `data/project_outputs/tables/`
- **Figures** under `data/project_outputs/figures/` or `reports/figures/`
- Any **checkpoints** or caches under `data/project_outputs/cache/`

The root **`RUBRIC_EVIDENCE_MAP.md`** aggregates notebook → artifact mapping for reviewers.
