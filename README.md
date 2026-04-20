# Leaf Disease Detection System

A web application that classifies plant leaf images into disease or healthy categories using convolutional and vision-transformer models trained on a multi-class setup (PlantVillage-style labels). After a prediction, optional text from a language model describes the predicted condition.

## Features

- **Web UI** — Upload an image, pick an architecture, and get a label, confidence score, and short description.
- **Multiple architectures** — ResNet50, InceptionV3, Hugging Face ViT, and a custom fine-tuned ViT (weights permitting).
- **Lazy loading** — At most **one** full model is kept in memory at a time to reduce RAM use on small hosts (for example, cloud free tiers).
- **38 classes** — Crops such as apple, tomato, potato, grape, and conditions from healthy to specific diseases (see the app UI for the full list).

## Project layout

| Path | Purpose |
|------|---------|
| `backend_app.py` | Flask app, inference routes, lazy loading logic |
| `app.py` | WSGI entry (`app:app`) for hosts that expect a module named `app` (e.g. Gunicorn on Render) |
| `templates/` | HTML templates for the web UI |
| `scripts/` | Training utilities, preprocessing, desktop demo (`app.py` tkinter), ViT helpers |
| `requirements.txt` | Python dependencies (PyTorch CPU wheels, Flask, Transformers, etc.) |
| `.python-version` | Suggests Python **3.12** for compatible wheels (e.g. Hugging Face `tokenizers` on Render) |

Large dataset folders (for example PlantVillage trees) and virtual environments are listed in `.gitignore`; do not commit bulky image datasets unless you intend to.

## Requirements

- **Python 3.12** (recommended; matches `.python-version` and typical cloud wheels).
- Dependencies are pinned in `requirements.txt` (Flask, Gunicorn, PyTorch/torchvision CPU builds, Transformers, Pillow, Requests).

Install:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Model weight files

Place these checkpoint files **next to `backend_app.py`** (repository root) if you want that architecture available:

| Model in UI | Expected filename |
|-------------|-------------------|
| ResNet50 | `best_resnet50_plant_disease_model_all_classes.pth` |
| InceptionV3 | `best_inceptionv3_plant_disease_model.pth` |
| ViT | `fine_tuned_vit_model.pth` |
| Custom ViT | `best_custom_vit_model.pth` |

The web UI only lists models whose files exist. ViT paths also download pretrained / Hugging Face weights on first load (network required).

## Run locally

```bash
python app.py
```

(or `python backend_app.py` — same app.)

Then open **http://127.0.0.1:5000** in your browser.

For a production-style local check:

```bash
gunicorn -c gunicorn.conf.py app:app
```

Use **one worker** (`-w 1`) so multiple processes do not each duplicate large models in memory.

## Deploy (e.g. Render)

- **Build:** `pip install -r requirements.txt`
- **Start (recommended):** `gunicorn -c gunicorn.conf.py app:app`  
  The repo’s **`gunicorn.conf.py`** binds to **`$PORT`**, uses **1 worker**, and sets **`timeout` to 600s** by default (override with env **`GUNICORN_TIMEOUT`**). Without this, Gunicorn’s **30s** default often kills workers during ViT / Hugging Face first load, which shows up as **HTTP 502** with an empty body.  
  The web UI also calls **`POST /prepare_model`** before **`POST /predict`**, so model download/load and inference are **separate requests**, each staying under the worker timeout when configured as above. If you still see 502s on ViT, the instance likely needs **more RAM** (OOM), not only a longer timeout.
- Set **Python 3.12** via `.python-version` or the host’s `PYTHON_VERSION` setting. Avoid default **3.14-only** environments if some packages lack wheels (build-from-source can fail on read-only build images).
- Ensure checkpoint `.pth` files are present in the deployment root (or adjust paths). Lazy loading helps **512 MiB** plans but a single large ViT can still require more RAM; upgrade the instance if you see out-of-memory errors.
- **Gemini descriptions:** add `GEMINI_API_KEY` in the dashboard (Environment / secrets). Never commit API keys to the repository.

## Disease descriptions (Gemini)

`scripts/desc_llm.py` calls Google’s **Generative Language API** for short disease summaries after each prediction.

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes, for live descriptions | Your [Google AI Studio / Gemini API](https://aistudio.google.com/apikey) key |
| `GEMINI_API_URL` | No | Full `…/models/…:generateContent` URL; defaults to **gemini-2.5-flash** if unset |

If `GEMINI_API_KEY` is missing, the app still returns predictions; the description field explains that the key is not configured.

**Local example (Windows PowerShell):**

```powershell
$env:GEMINI_API_KEY = "your-key-here"
python backend_app.py
```

**Local example (macOS / Linux):**

```bash
export GEMINI_API_KEY="your-key-here"
python backend_app.py
```

## Training and experimentation

The `scripts/` directory contains trainers (for example ResNet, Inception, ViT transfer learning), preprocessing, and notebooks—useful if you retrain or evaluate on your own splits of plant disease data.

## References

- [PlantVillage dataset](https://arxiv.org/abs/1511.08060) (classic reference for plant disease classification)
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/) / [TorchVision](https://pytorch.org/vision/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
