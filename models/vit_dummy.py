import cv2
import numpy as np
import os

# Simple heuristic-based pupil detector.
# Replace this with your trained ViT model inference later.

def predict_eye(image_path, output_folder='static/results'):
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'reason': 'Cannot read image'}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalize + blur helps with lighting
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Threshold to find dark areas (pupil tends to be dark)
    _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'success': False, 'reason': 'No dark contours found'}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    if radius <= 2:
        return {'success': False, 'reason': 'Detected region too small'}

    # create mask (single-channel)
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)

    # overlay (color)
    overlay = img.copy()
    overlay[mask == 255] = (0, 0, 255)  # red tint for pupil
    alpha = 0.5
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    mask_fn = f"{name}_mask.png"
    overlay_fn = f"{name}_overlay.png"
    mask_path = os.path.join(output_folder, mask_fn)
    overlay_path = os.path.join(output_folder, overlay_fn)

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, out)

    pupil_diameter_px = radius * 2
    # Very rough heuristic for "abnormal" pupil: outside typical range
    cvs_flag = (pupil_diameter_px < 30) or (pupil_diameter_px > 120)
    confidence = min(0.95, abs(pupil_diameter_px - 60) / 100 + 0.5)

    return {
        'success': True,
        'mask': mask_fn,
        'overlay': overlay_fn,
        'pupil_px': int(pupil_diameter_px),
        'area_px': int(area),
        'cvs_flag': bool(cvs_flag),
        'confidence': float(confidence)
    }
