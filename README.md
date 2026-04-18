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
python backend_app.py
```

Then open **http://127.0.0.1:5000** in your browser.

For a production-style local check:

```bash
gunicorn app:app --bind 127.0.0.1:5000 -w 1
```

Use **one worker** (`-w 1`) so multiple processes do not each duplicate large models in memory.

## Deploy (e.g. Render)

- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT -w 1`
- Set **Python 3.12** via `.python-version` or the host’s `PYTHON_VERSION` setting. Avoid default **3.14-only** environments if some packages lack wheels (build-from-source can fail on read-only build images).
- Ensure checkpoint `.pth` files are present in the deployment root (or adjust paths). Lazy loading helps **512 MiB** plans but a single large ViT can still require more RAM; upgrade the instance if you see out-of-memory errors.

## Optional: disease descriptions

`scripts/desc_llm.py` calls the Google Gemini API for short text descriptions. For production, keep API keys out of source control and load them from environment variables or your host’s secret store.

## Training and experimentation

The `scripts/` directory contains trainers (for example ResNet, Inception, ViT transfer learning), preprocessing, and notebooks—useful if you retrain or evaluate on your own splits of plant disease data.

## References

- [PlantVillage dataset](https://arxiv.org/abs/1511.08060) (classic reference for plant disease classification)
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/) / [TorchVision](https://pytorch.org/vision/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
