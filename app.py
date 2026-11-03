import base64
import time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from enhancement import enhance_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_IMAGE_PIXELS = 20_000_000

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def strength(name, default):
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {name}.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/enhance")
def enhance():
    if "image" not in request.files:
        return jsonify(error="Please choose an image first."), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify(error="Please choose an image first."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Use PNG, JPG, BMP or TIFF images."), 400
    raw = file.read()
    if not raw:
        return jsonify(error="The selected file is empty."), 400
    try:
        original = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if original is None or original.size == 0:
            return jsonify(error="OpenCV could not read this image."), 400
        height, width = original.shape[:2]
        if width * height > MAX_IMAGE_PIXELS:
            return jsonify(error="Image is too large. Maximum is 20 megapixels."), 400

        brightness = strength("brightness", 0.55)
        contrast = strength("contrast", 0.45)
        denoise = strength("denoise", 0.25)
        sharpen = strength("sharpen", 0.20)
        preset = int(request.form.get("preset", "0"))
        if preset not in (0, 1, 2):
            raise ValueError("Invalid preset.")

        started = time.perf_counter()
        result = enhance_image(original, brightness, contrast, denoise, sharpen, preset)
        elapsed = time.perf_counter() - started
        if result is None or result.size == 0:
            raise RuntimeError("Enhancement returned an empty image.")

        fmt = request.form.get("format", "png").lower()
        if fmt == "jpeg":
            ok, output = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mime, extension = "image/jpeg", "jpg"
        else:
            ok, output = cv2.imencode(".png", result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            mime, extension = "image/png", "png"
        if not ok:
            raise RuntimeError("Could not encode the enhanced image.")
        image_data = base64.b64encode(output.tobytes()).decode("ascii")
        return jsonify(
            image=f"data:{mime};base64,{image_data}",
            filename=f"lumina_enhanced.{extension}",
            width=width, height=height,
            processing_seconds=round(elapsed, 3)
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Image enhancement failed")
        return jsonify(error="Enhancement failed. Try another image or adjust the settings."), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="File is too large. Maximum upload size is 12 MB."), 413

@app.get("/health")
def health():
    return jsonify(status="ok", application="LUMINA")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
