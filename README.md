# Fast Image Captioner (BLIP)

This project uses **Salesforce/blip-image-captioning-base** for faster image captioning compared to ViT-GPT2 models.

## Files
- `inference.py` — BLIP-based captioner (auto-selects GPU if available)
- `app_gradio.py` — Gradio UI (recommended for Hugging Face Spaces)
- `app_streamlit.py` — Streamlit UI (for Streamlit Community Cloud)
- `app.py` — wrapper entrypoint for HF Spaces
- `run_cli.py` — simple CLI to caption local images
- `requirements.txt` — minimal dependencies

## Quick Colab steps (manual upload)
1. Open Google Colab → New Notebook.
2. Upload the ZIP via the file picker (run `from google.colab import files; files.upload()`).
3. Unzip and install:
```bash
!unzip -o image-captioner-fast.zip -d image-captioner-fast
%cd image-captioner-fast
!pip install -r requirements.txt
```
4. Run Gradio (temporary public link):
```bash
# in a cell
!python app.py
```
Gradio will print a public `.gradio.live` URL while the Colab session runs.

## Deploy
- **Hugging Face Spaces**: Create a Gradio Space and push this repo (ensure `app.py` and `requirements.txt` are at repo root).
- **Streamlit Community Cloud**: Push to GitHub and deploy `app_streamlit.py`.

## Tips
- Model weights are downloaded on first run (~200-400 MB depending on model).
- Use GPU (Colab GPU runtime) for faster inference.
- Do NOT commit large model files to GitHub/Spaces; rely on HF Hub model loading.

Enjoy! ⚡
