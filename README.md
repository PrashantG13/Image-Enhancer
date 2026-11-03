# Low-Light Image Enhancement (Web)

Flask web version of the supplied desktop project. The original classical enhancement functions in `enhancement.py` are preserved unchanged.

## Features
- Upload PNG, JPG/JPEG, BMP and TIFF (12 MB maximum).
- Original/enhanced previews.
- Gamma correction, luminance-based CLAHE, Non-Local Means colour denoising and unsharp masking.
- Three presets and four adjustment controls.
- Download enhanced output as PNG or JPEG.

## Run locally (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Render deployment

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- No environment variables are required.
- `Procfile` is included: `web: gunicorn app:app`

## Pipeline

The supplied `enhancement.py` is copied unchanged. It performs:
1. Gray-level power-law / gamma transformation.
2. CLAHE on the LAB luminance channel.
3. OpenCV coloured Non-Local Means denoising.
4. Unsharp-mask detail enhancement.

This is a classical image-processing application; it does not train a neural network. Uploaded images are processed in memory and are not saved permanently by this app.
