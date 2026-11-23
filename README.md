# Skin Lesion CNN — Streamlit Demo

This repository packages the supplied CNN model in a Streamlit application.

## Important limitation

The supplied training script (`cnn_skin_cancer.py`) creates its binary labels using
the median average brightness of each image:
- Class 0: darker image
- Class 1: lighter image

These are synthetic brightness labels, not dermatologist-confirmed diagnoses or
verified benign/malignant labels. Consequently, this application is only a software
demonstration of the model's darker/lighter image classification. It must not be
described or used as a skin cancer diagnostic tool.

## Repository files

- `app.py` — Streamlit user interface and inference code.
- `isic_skin_lesion_model.keras` — trained Keras model used by the app.
- `isic_skin_lesion_model.pkl` — supplied pickle copy, included for archival purposes only; the app does not load it.
- `cnn_skin_cancer.py` — supplied training/source script.
- `cnn_skin_cancer.ipynb` — supplied notebook.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes local environment and temporary files.

The `.pkl` copy is not needed by the app. The app loads the `.keras` model directly.
Avoid loading pickle files from untrusted sources.

## Run locally

Use Python 3.11 for this deployment setup.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Upload to GitHub

1. Sign in to GitHub and create a **new repository**.
2. Choose a repository name, for example `skin-lesion-cnn-streamlit`.
3. Set it to Public if you want Streamlit Community Cloud to deploy it without
   private-repository access setup. Do not upload private patient images or data.
4. In the repository's **Add file → Upload files** page, upload the project files
   from this folder. Keep `app.py`, `requirements.txt`, and
   `isic_skin_lesion_model.keras` in the repository root.
5. Commit the files to the `main` branch.

The model file is about 2.5 MB, so Git LFS should not be necessary for this file.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using your GitHub account.
2. Select **Create app**.
3. Choose the repository you just created, branch `main`, and main file path
   `app.py`.
4. Open **Advanced settings** if you need to set the Python version; choose
   Python 3.11 if the option is shown.
5. Select **Deploy** and wait for dependency installation and startup.
6. Open the generated app URL to test it.

If deployment fails, open the app's **Manage app → Logs** view and check the first
dependency or file-path error. Confirm that the model file is committed beside
`app.py` and that `requirements.txt` is in the repository root.

## Notes about the original training code

The supplied training script uses `/content/` paths from Google Colab and expects
`ISIC_*.jpg` images to be available. Those dataset images are not included in this
deployment package. The Streamlit app does not retrain the model; it loads the
provided `.keras` file for inference.
