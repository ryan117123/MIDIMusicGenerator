# Attribution

This document records **datasets**, **external resources**, and **AI-assisted development** used in this project, per course integrity requirements.

---

## Dataset

| Item | Details |
|------|---------|
| **Name** | Piano MIDI corpus distributed via Google Magenta (commonly referred to as the MAESTRO dataset distribution). |
| **Source** | https://magenta.tensorflow.org/datasets/maestro |
| **Use in this project** | Training and evaluation of next-token models on aligned piano performances; metadata and MIDI paths referenced in `notebooks/01_data_audit_preprocess.ipynb` and downstream notebooks. |
| **License / terms** | Use is subject to the **dataset license and citation requirements** on the official page above. **You must** verify the current license before submission and cite the dataset in any written report or video. |

No other external **datasets** were required for the core pipeline beyond what you place under `data/raw/` and process into `data/project_outputs/`.

---

## External code and libraries

| Item | Details |
|------|---------|
| **Python ecosystem** | Standard open-source libraries listed in **`requirements.txt`** (e.g. NumPy, pandas, PyTorch, Gradio, `pretty_midi`, scikit-learn, Jupyter). |
| **Third-party snippets** | No standalone copy-pasted code blocks from third-party tutorials were incorporated without attribution beyond normal library usage. If you add such snippets later, append a row here with source URL and license. |

---

## AI development assistance (Cursor)

| Item | Details |
|------|---------|
| **Tool** | **Cursor** (IDE), including **Cursor Agent** / AI-assisted editing features used during development. |
| **Scope** | Cursor and Cursor agents were used to help create and refine **much of the codebase and documentation** for this repository—including but not limited to: Python scripts (e.g. `scripts/gradio_demo.py`), Jupyter notebooks, `README.md`, `SETUP.md`, `requirements.txt`, and this `ATTRIBUTION.md` file. |
| **Human role** | The author **reviewed**, **ran**, and **debugged** generated material; chose **models, hyperparameters, and experiments**; curated **data and results**; and is responsible for the **accuracy of claims** in reports and demos. |
| **Other AI tools** | If you used additional tools (e.g. ChatGPT, Copilot outside Cursor), add a subsection below with tool name, scope, and your review process. |

