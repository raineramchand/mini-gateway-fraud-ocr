import os
import json
import re
import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import easyocr

# Logger for debugging
logger = logging.getLogger("uvicorn.error")

# Initialize EasyOCR reader (set gpu=True if you have a GPU available)
reader = easyocr.Reader(['en'], gpu=False)

# ===================== IMAGE PROCESSING =====================

def detect_printed_area(image: np.ndarray) -> np.ndarray:
    """
    Crop the image to the largest printed area using contour detection.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = 20
            return image[max(0, y-pad): y+h+pad,
                         max(0, x-pad): x+w+pad]
        return image
    except Exception:
        logger.exception("Error in detect_printed_area")
        return image


def correct_skew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew in a grayscale image using Hough transform.
    """
    try:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for rho, theta in lines[:10]:
                angle = theta * 180 / np.pi
                if angle > 45:
                    angle -= 90
                angles.append(angle)
            angle = np.median(angles)
            if abs(angle) > 0.5:
                return ndimage.rotate(image, angle, reshape=False, cval=255)
        return image
    except Exception:
        logger.exception("Error in correct_skew")
        return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, deskew, enhance, denoise, and clean the image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input to preprocess_image must be a numpy array")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    deskewed = correct_skew(gray)
    enhanced = cv2.equalizeHist(deskewed)
    denoised = cv2.medianBlur(enhanced, 3)
    cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
    return cleaned

# ===================== FIELD EXTRACTION HELPERS =====================

MERCHANT_KEYWORDS = [
    'TRADER JOE', 'WALMART', 'WHOLE FOODS', 'COSTCO', 'SAFEWAY', 'KROGER',
    'TARGET', 'CVS', 'WALGREENS', 'MCDONALD', 'STARBUCKS', 'SUBWAY',
    'SPAR', 'WINCO', 'MOMI', 'TOY', 'STORE', 'MARKET', 'SHOP'
]


def group_text_lines(text_data: List[Tuple[List[List[float]], str, float]]) -> List[str]:
    """
    Group OCR output into lines based on y-coordinate proximity.
    Returns a list of merged text lines, in natural top-down order.
    """
    # text_data items: (bbox, text, confidence)
    lines = []
    for bbox, text, conf in text_data:
        # average y of bbox
        ys = [point[1] for point in bbox]
        y_mean = sum(ys) / len(ys)
        lines.append((y_mean, text.strip()))
    # sort by y
    lines.sort(key=lambda x: x[0])

    merged = []
    current_y, buffer = None, []
    for y, txt in lines:
        if current_y is None or abs(y - current_y) < 8:
            buffer.append(txt)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            merged.append(" ".join(buffer))
            buffer = [txt]
            current_y = y
    if buffer:
        merged.append(" ".join(buffer))
    return merged


def extract_merchant_name(text_lines: List[str]) -> Optional[str]:
    """
    Scan the top lines for known merchant keywords or default to the first line.
    """
    for i, line in enumerate(text_lines[:5]):
        upper = line.upper()
        if len(upper) < 3 or re.fullmatch(r"\d+[\d\s\-/]*", upper):
            continue
        for keyword in MERCHANT_KEYWORDS:
            if keyword in upper:
                return re.sub(r"[^\w\s&'-]", ' ', line).strip()
        if i == 0:
            return re.sub(r"[^\w\s&'-]", ' ', line).strip()
    return None


def extract_total_easyocr(text_lines: List[str]) -> Optional[float]:
    """
    1) Look for a line containing 'TOTAL' and extract the first price on that line.
    2) Fallback: extract all prices and return the maximum.
    """
    # 1. TOTAL line
    for line in text_lines:
        if re.search(r"\bTOTAL\b", line, re.IGNORECASE):
            m = re.search(r"\$?(\d+\.\d{2})", line)
            if m:
                return float(m.group(1))
    # 2. Fallback to max price in all lines
    all_amounts = []
    for line in text_lines:
        found = re.findall(r"\$?(\d+\.\d{2})", line)
        all_amounts.extend(map(float, found))
    return max(all_amounts) if all_amounts else None


def extract_receipt_info(image_path: str) -> Tuple[str, float, str]:
    """
    Run OCR on the receipt image and return (merchant_name, total, engine_used).

    Raises:
        FileNotFoundError: if the image_path does not exist.
        ValueError: if merchant or total cannot be extracted.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Receipt image not found: {image_path}")

    # Load and preprocess
    image = np.array(Image.open(image_path))
    cropped = detect_printed_area(image)
    processed = preprocess_image(cropped)

    # OCR via EasyOCR
    ocr_data = reader.readtext(processed)
    text_lines = group_text_lines(ocr_data)

    merchant = extract_merchant_name(text_lines)
    total = extract_total_easyocr(text_lines)
    engine = "easyocr"

    if merchant is None or total is None:
        raise ValueError(f"OCR failed: merchant={merchant}, total={total}")

    logger.debug("Extracted merchant=%r, total=%.2f, engine=%s", merchant, total, engine)
    return merchant, total, engine


def process_folder(folder_path: str = "images") -> dict:
    results = {}
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(folder_path, filename)
        try:
            merchant, total, engine = extract_receipt_info(path)
        except Exception as e:
            logger.error("Failed OCR for %s: %s", filename, e)
            merchant, total, engine = None, None, f"error: {e}"
        results[filename] = {
            "merchant_name": merchant,
            "total_amount": total,
            "ocr_engine_used": engine
        }
        logger.info("%s -> merchant=%r, total=%r, engine=%s", filename, merchant, total, engine)

    with open("ocr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
