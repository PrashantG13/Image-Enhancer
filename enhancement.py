import cv2
import numpy as np


def _to_uint8(image):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    return np.ascontiguousarray(image)


def gamma_transform(image, brightness=0.55):
    """Gray-level power-law transform. Lower gamma brightens dark pixels."""
    brightness = float(np.clip(brightness, 0.0, 1.0))
    # Map slider strength to gamma in a useful range.
    gamma = 1.65 - (brightness * 1.25)
    gamma = max(0.40, gamma)

    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def histogram_equalization(image, strength=0.45, clip_limit=2.0):
    """Apply CLAHE to luminance only to avoid changing natural colors."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image.copy()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(np.clip(clip_limit, 1.0, 4.0)),
        tileGridSize=(8, 8)
    )
    equalized = clahe.apply(l_channel)

    # Blend with original luminance to prevent an over-processed look.
    blended_l = cv2.addWeighted(
        l_channel, 1.0 - strength,
        equalized, strength, 0
    )

    result = cv2.merge((blended_l, a_channel, b_channel))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def denoise_filter(image, strength=0.25):
    """Edge-preserving denoising; avoids excessive blur at normal settings."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image.copy()

    h = 2.0 + strength * 7.0
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, 7, 21
    )


def unsharp_mask(image, strength=0.20):
    """Mild detail sharpening with protection against strong halos."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), 1.15)
    amount = 0.15 + strength * 0.75
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_image(
    image,
    brightness=0.55,
    contrast=0.45,
    denoise=0.25,
    sharpen=0.20,
    preset=0
):
    """
    Complete classical low-light enhancement pipeline.

    brightness, contrast, denoise and sharpen are values from 0.0 to 1.0.
    preset: 0 balanced, 1 natural/subtle, 2 strong recovery.
    """
    original = _to_uint8(image)

    if preset == 1:  # Natural / subtle
        brightness = min(brightness, 0.48)
        contrast = min(contrast, 0.35)
        denoise = min(denoise, 0.25)
        sharpen = min(sharpen, 0.20)
    elif preset == 2:  # Strong low-light recovery
        brightness = min(1.0, brightness + 0.16)
        contrast = min(1.0, contrast + 0.12)
        denoise = min(1.0, denoise + 0.10)

    # Step 1: gray-level power-law transformation.
    result = gamma_transform(original, brightness)

    # Step 2: luminance-only adaptive histogram equalization.
    result = histogram_equalization(
        result,
        strength=contrast,
        clip_limit=2.0 + contrast * 1.2
    )

    # Step 3: edge-preserving noise filtering.
    result = denoise_filter(result, denoise)

    # Step 4: restrained high-frequency detail enhancement.
    result = unsharp_mask(result, sharpen)

    return np.ascontiguousarray(result)
