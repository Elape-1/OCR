from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

logger = logging.getLogger(__name__)
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# ---------------------------------------------------------------------------
# OCR quality-gate constants (Priorities 1-3)
# ---------------------------------------------------------------------------
MIN_OCR_CONFIDENCE: int = 25
"""Drop tokens with Tesseract confidence below this value (0-100)."""

MIN_PAGE_TOKENS: int = 3
"""Pages yielding fewer tokens than this after filtering are treated as blank."""

BLANK_PAGE_CONFIDENCE: float = 30.0
"""If all surviving tokens are below this confidence, treat the page as blank."""


def _configure_tesseract_cmd() -> None:
    candidates: list[str] = []
    if TESSERACT_CMD:
        candidates.append(TESSERACT_CMD)

    detected_on_path = shutil.which("tesseract")
    if detected_on_path:
        candidates.append(detected_on_path)

    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
            ]
        )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            logger.info("Using Tesseract binary at %s", candidate)
            return


_configure_tesseract_cmd()


def preprocess_page_image(image_path: str | Path) -> np.ndarray:
    processed, _ = _preprocess_page_image_with_transform(image_path)
    return processed


def _preprocess_page_image_with_transform(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    original_height, original_width = image.shape[:2]
    rotation = _detect_orientation_rotation(image)
    corrected = _correct_orientation(image)
    orientation_transform = _orientation_transform(rotation, original_width, original_height)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    deskewed, deskew_transform = _deskew_with_transform(thresholded)
    source_to_processed = np.vstack((deskew_transform, [0.0, 0.0, 1.0])) @ np.vstack(
        (orientation_transform, [0.0, 0.0, 1.0])
    )
    processed_to_source = np.linalg.inv(source_to_processed).astype(np.float32)
    return deskewed, processed_to_source[:2], (original_width, original_height)


def extract_tokens_with_bboxes(image_path: str | Path) -> List[Dict[str, Any]]:
    processed, processed_to_source, (source_width, source_height) = _preprocess_page_image_with_transform(image_path)
    height, width = processed.shape[:2]
    try:
        data = pytesseract.image_to_data(
            processed,
            output_type=Output.DICT,
            config="--oem 1 --psm 3",
            lang="eng",
        )

        tokens: List[Dict[str, Any]] = []
        for index, text in enumerate(data["text"]):
            token = text.strip()
            if not token:
                continue
            confidence = float(data["conf"][index]) if data["conf"][index] not in {"-1", -1} else 0.0
            processed_bbox = [
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["left"][index] + data["width"][index]),
                int(data["top"][index] + data["height"][index]),
            ]
            absolute_bbox = _map_bbox_to_source(processed_bbox, processed_to_source, source_width, source_height)
            tokens.append(
                {
                    "token": token,
                    "bbox": normalize_bbox(absolute_bbox, source_width, source_height),
                    "absolute_bbox": absolute_bbox,
                    "confidence": confidence,
                    "block_num": int(data["block_num"][index]),
                    "paragraph_num": int(data["par_num"][index]),
                    "line_num": int(data["line_num"][index]),
                    "word_num": int(data["word_num"][index]),
                }
            )

        # -- Priority 2: confidence threshold filtering -----------------
        pre_filter_count = len(tokens)
        tokens = [t for t in tokens if t["confidence"] >= MIN_OCR_CONFIDENCE]
        filtered_count = pre_filter_count - len(tokens)
        if filtered_count > 0:
            logger.debug(
                "Filtered %d low-confidence tokens (<%d) from %s",
                filtered_count,
                MIN_OCR_CONFIDENCE,
                image_path,
            )

        # -- Priority 3: blank / near-blank page detection --------------
        if len(tokens) < MIN_PAGE_TOKENS:
            has_boundary_token = any(t["confidence"] == MIN_OCR_CONFIDENCE for t in tokens)
            if not tokens or (
                all(t["confidence"] < BLANK_PAGE_CONFIDENCE for t in tokens)
                and not has_boundary_token
            ):
                logger.info(
                    "Page detected as blank/near-blank (%d tokens after filter): %s",
                    len(tokens),
                    image_path,
                )
                return []

        return tokens
    except Exception as exc:
        logger.warning("Tesseract OCR failed for %s; using fallback token extraction: %s", image_path, exc)
        return _fallback_tokens(processed, width, height, processed_to_source, source_width, source_height)

def _fallback_tokens(
    processed: np.ndarray,
    width: int,
    height: int,
    processed_to_source: np.ndarray | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
) -> List[Dict[str, Any]]:
    binary = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    tokens: List[Dict[str, Any]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 20:
            continue
        processed_bbox = [int(x), int(y), int(x + w), int(y + h)]
        if processed_to_source is not None and source_width is not None and source_height is not None:
            absolute_bbox = _map_bbox_to_source(processed_bbox, processed_to_source, source_width, source_height)
            bbox_width, bbox_height = source_width, source_height
        else:
            absolute_bbox = processed_bbox
            bbox_width, bbox_height = width, height
        tokens.append(
            {
                "token": "fallback",
                "bbox": normalize_bbox(absolute_bbox, bbox_width, bbox_height),
                "absolute_bbox": absolute_bbox,
                "confidence": 0.0,
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 1,
                "word_num": label,
            }
        )
    return tokens


def _map_bbox_to_source(
    bbox: List[int], processed_to_source: np.ndarray, source_width: int, source_height: int
) -> List[int]:
    x0, y0, x1, y1 = bbox
    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, processed_to_source).reshape(-1, 2)
    return [
        max(0, min(source_width, int(np.floor(mapped[:, 0].min() + 1e-5)))),
        max(0, min(source_height, int(np.floor(mapped[:, 1].min() + 1e-5)))),
        max(0, min(source_width, int(np.ceil(mapped[:, 0].max() - 1e-5)))),
        max(0, min(source_height, int(np.ceil(mapped[:, 1].max() - 1e-5)))),
    ]


def normalize_bbox(bbox: List[int], page_width: int, page_height: int) -> List[int]:
    x0, y0, x1, y1 = bbox
    if page_width <= 0 or page_height <= 0:
        raise ValueError("Page dimensions must be positive for bounding box normalization")

    scaled = [
        _clamp(round(1000 * x0 / page_width)),
        _clamp(round(1000 * y0 / page_height)),
        _clamp(round(1000 * x1 / page_width)),
        _clamp(round(1000 * y1 / page_height)),
    ]
    return scaled


def _clamp(value: float | int) -> int:
    return int(max(0, min(1000, value)))


def _correct_orientation(image: np.ndarray) -> np.ndarray:
    rotation = _detect_orientation_rotation(image)
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _detect_orientation_rotation(image: np.ndarray) -> int:
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        rotation = 0
        for line in osd.splitlines():
            if line.startswith("Rotate:"):
                rotation = int(line.split(":", 1)[1].strip())
                break
        return rotation if rotation in {0, 90, 180, 270} else 0
    except Exception:
        logger.debug("Orientation detection failed, continuing without rotation", exc_info=True)
    return 0


def _orientation_transform(rotation: int, width: int, height: int) -> np.ndarray:
    if rotation == 90:
        return np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    if rotation == 180:
        return np.array([[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0]], dtype=np.float32)
    if rotation == 270:
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, width - 1.0]], dtype=np.float32)
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _deskew(image: np.ndarray) -> np.ndarray:
    rotated, _ = _deskew_with_transform(image)
    return rotated


def _deskew_with_transform(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.column_stack(np.where(image < 255))
    if coordinates.size == 0:
        return image, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1:
        return image, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, rotation_matrix.astype(np.float32)
