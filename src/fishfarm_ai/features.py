from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageFeatureSpec:
    size: tuple[int, int] = (128, 128)
    color_bins: int = 8
    orientation_bins: int = 4


SPEC = ImageFeatureSpec()


def extract_image_features(image: Image.Image) -> np.ndarray:
    """Extract compact biological visual descriptors from a fish image.

    Captures:
    1. Global RGB and HSV color statistics (means and standard deviations).
    2. Dorsal vs Ventral color differential (dark back vs pale belly, crucial for siluroid fish).
    3. Quantized HSV and RGB color distribution histograms.
    4. Multi-directional edge gradient statistics and orientation histograms.
    5. Laplacian texture energy (discriminating fish scales from smooth mucous skin).
    6. Foreground contour morphology (body aspect ratio and silhouette compactness).
    """
    rgb_img = image.convert("RGB").resize(SPEC.size)
    arr_rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
    arr_u8 = (arr_rgb * 255.0).astype(np.uint8)

    gray = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

    features: list[float] = []

    # 1. Global Color stats (RGB & HSV)
    features.extend(arr_rgb.mean(axis=(0, 1)).tolist())
    features.extend(arr_rgb.std(axis=(0, 1)).tolist())
    features.extend(hsv.mean(axis=(0, 1)).tolist())
    features.extend(hsv.std(axis=(0, 1)).tolist())

    # 2. Dorsal vs Ventral color difference (top half vs bottom half)
    top_half = arr_rgb[: SPEC.size[1] // 2, :, :]
    bot_half = arr_rgb[SPEC.size[1] // 2 :, :, :]
    dv_diff = (top_half.mean(axis=(0, 1)) - bot_half.mean(axis=(0, 1))).tolist()
    features.extend(dv_diff)

    # 3. Compact Color histograms (HSV & RGB)
    for ch in range(3):
        h, _ = np.histogram(hsv[:, :, ch], bins=SPEC.color_bins, range=(0.0, 1.0), density=True)
        features.extend(h.astype(np.float32).tolist())
    for ch in range(3):
        h, _ = np.histogram(arr_rgb[:, :, ch], bins=SPEC.color_bins, range=(0.0, 1.0), density=True)
        features.extend(h.astype(np.float32).tolist())

    # 4. Multi-directional gradients (Sobel) and edge orientation
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    features.append(float(mag.mean()))
    features.append(float(mag.std()))
    features.append(float((mag > mag.mean()).mean()))

    # Edge orientation histogram
    ang_deg = ((angle + np.pi) * 180.0 / np.pi) % 180.0
    if float(mag.sum()) > 1e-6:
        ang_hist, _ = np.histogram(
            ang_deg,
            bins=SPEC.orientation_bins,
            range=(0.0, 180.0),
            weights=mag,
            density=True,
        )
        features.extend(ang_hist.astype(np.float32).tolist())
    else:
        features.extend([1.0 / float(SPEC.orientation_bins)] * SPEC.orientation_bins)

    # 5. Texture scale energy (Laplacian variance) - distinguishes scales vs smooth skin
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    features.append(float(lap.var()))
    features.append(float(np.abs(lap).mean()))

    # 6. Shape & Elongation from Otsu foreground silhouette
    gray_u8 = (gray * 255).astype(np.uint8)
    _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c) / float(SPEC.size[0] * SPEC.size[1])
        _, _, w, h = cv2.boundingRect(c)
        aspect = float(w) / max(float(h), 1.0)
        extent = cv2.contourArea(c) / max(float(w * h), 1.0)
        features.extend([float(area), float(aspect), float(extent)])
    else:
        features.extend([0.5, 1.0, 0.5])

    return np.asarray(features, dtype=np.float32)
