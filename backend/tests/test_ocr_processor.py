"""Unit tests for ocr_processor OCR quality-gate changes (Priorities 1-3).

These tests mock pytesseract so they run without Tesseract installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_image(width: int = 800, height: int = 1100) -> np.ndarray:
    """Create a simple grayscale test image with dark text on white."""
    img = np.full((height, width), 255, dtype=np.uint8)
    cv2.putText(img, "HELLO WORLD", (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5, cv2.LINE_AA)
    return img


def _make_tesseract_dict(
    texts: list[str],
    confs: list[int | float],
    *,
    lefts: list[int] | None = None,
    tops: list[int] | None = None,
    widths: list[int] | None = None,
    heights: list[int] | None = None,
) -> dict:
    """Build a dict matching pytesseract.image_to_data(output_type=DICT)."""
    n = len(texts)
    if lefts is None:
        lefts = [10 + i * 80 for i in range(n)]
    if tops is None:
        tops = [100] * n
    if widths is None:
        widths = [70] * n
    if heights is None:
        heights = [20] * n
    return {
        "text": texts,
        "conf": confs,
        "left": lefts,
        "top": tops,
        "width": widths,
        "height": heights,
        "block_num": [1] * n,
        "par_num": [1] * n,
        "line_num": [1] * n,
        "word_num": list(range(1, n + 1)),
    }


# ---------------------------------------------------------------------------
# Priority 1: PSM 3
# ---------------------------------------------------------------------------


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_psm3_config_used(mock_tess, _mock_orient, tmp_path):
    """Verify that Tesseract is called with --psm 3 instead of --psm 6."""
    from app.ocr_processor import extract_tokens_with_bboxes

    # Write a test image so cv2.imread succeeds
    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["Hello", "World"], [95, 90],
    )
    mock_tess.Output = MagicMock()

    extract_tokens_with_bboxes(str(img_path))

    call_args = mock_tess.image_to_data.call_args
    config_arg = call_args.kwargs.get("config") or call_args[1].get("config", "")
    assert "--psm 3" in config_arg, f"Expected --psm 3 in config, got: {config_arg}"
    assert "--psm 6" not in config_arg


# ---------------------------------------------------------------------------
# Priority 2: Confidence filtering
# ---------------------------------------------------------------------------


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_confidence_filter_drops_low_tokens(mock_tess, _mock_orient, tmp_path):
    """Tokens with confidence < MIN_OCR_CONFIDENCE (25) should be removed."""
    from app.ocr_processor import extract_tokens_with_bboxes, MIN_OCR_CONFIDENCE

    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    # Mix of high and low confidence tokens
    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["Good", "Noise", "OK", "Garbage", "Fine"],
        [90, 10, 80, 5, 60],
    )
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes(str(img_path))
    token_texts = [t["token"] for t in tokens]

    assert "Good" in token_texts
    assert "OK" in token_texts
    assert "Fine" in token_texts
    assert "Noise" not in token_texts
    assert "Garbage" not in token_texts


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_confidence_filter_keeps_boundary_tokens(mock_tess, _mock_orient, tmp_path):
    """Tokens at exactly MIN_OCR_CONFIDENCE should survive."""
    from app.ocr_processor import extract_tokens_with_bboxes, MIN_OCR_CONFIDENCE

    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["Boundary", "Above", "Below"],
        [MIN_OCR_CONFIDENCE, MIN_OCR_CONFIDENCE + 1, MIN_OCR_CONFIDENCE - 1],
    )
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes(str(img_path))
    token_texts = [t["token"] for t in tokens]

    assert "Boundary" in token_texts  # exactly at threshold → kept
    assert "Above" in token_texts
    assert "Below" not in token_texts


# ---------------------------------------------------------------------------
# Priority 3: Blank page detection
# ---------------------------------------------------------------------------


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_blank_page_returns_empty(mock_tess, _mock_orient, tmp_path):
    """A page with only 1-2 low-confidence tokens should return []."""
    from app.ocr_processor import extract_tokens_with_bboxes

    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    # Two tokens, both below BLANK_PAGE_CONFIDENCE (30) but above MIN_OCR_CONFIDENCE (25)
    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["a", "b"],
        [26, 27],
    )
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes(str(img_path))
    assert tokens == [], f"Expected empty list for blank page, got {len(tokens)} tokens"


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_blank_page_all_filtered(mock_tess, _mock_orient, tmp_path):
    """A page where ALL tokens are below MIN_OCR_CONFIDENCE should return []."""
    from app.ocr_processor import extract_tokens_with_bboxes

    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["x", "y", "z"],
        [5, 10, 3],
    )
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes(str(img_path))
    assert tokens == []


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_sparse_valid_page_not_blanked(mock_tess, _mock_orient, tmp_path):
    """A page with few tokens but good confidence should NOT be treated as blank."""
    from app.ocr_processor import extract_tokens_with_bboxes

    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    # Only 2 tokens but one has confidence >= BLANK_PAGE_CONFIDENCE
    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["Chapter", "1"],
        [85, 90],
    )
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes(str(img_path))
    assert len(tokens) == 2, "Sparse but valid page should keep its tokens"
    assert tokens[0]["token"] == "Chapter"


@patch("app.ocr_processor._correct_orientation", side_effect=lambda img: img)
@patch("app.ocr_processor.pytesseract")
def test_normal_page_unaffected(mock_tess, _mock_orient, tmp_path):
    """A normal page with many high-confidence tokens passes through unchanged."""
    from app.ocr_processor import extract_tokens_with_bboxes

    img = _make_test_image()
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    texts = [f"word{i}" for i in range(20)]
    confs = [85 + (i % 10) for i in range(20)]
    mock_tess.image_to_data.return_value = _make_tesseract_dict(texts, confs)
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes(str(img_path))
    assert len(tokens) == 20


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_min_ocr_confidence_constant():
    """Verify the confidence threshold constant is importable and correct."""
    from app.ocr_processor import MIN_OCR_CONFIDENCE
    assert MIN_OCR_CONFIDENCE == 25


def test_rotated_box_maps_back_to_source_coordinates():
    from app.ocr_processor import _map_bbox_to_source, _orientation_transform

    transform = _orientation_transform(90, width=100, height=200)
    source_bbox = [10, 20, 30, 40]
    processed_bbox = [199 - source_bbox[3], source_bbox[0], 199 - source_bbox[1], source_bbox[2]]

    assert _map_bbox_to_source(processed_bbox, np.linalg.inv(
        np.vstack((transform, [0.0, 0.0, 1.0]))
    )[:2], 100, 200) == source_bbox


@patch("app.ocr_processor._preprocess_page_image_with_transform")
@patch("app.ocr_processor.pytesseract")
def test_ocr_boxes_are_normalized_to_source_dimensions(mock_tess, mock_preprocess):
    from app.ocr_processor import extract_tokens_with_bboxes

    processed = np.full((100, 200), 255, dtype=np.uint8)
    mock_preprocess.return_value = (
        processed,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (400, 200),
    )
    mock_tess.image_to_data.return_value = _make_tesseract_dict(
        ["source"], [95], lefts=[20], tops=[10], widths=[20], heights=[20]
    )
    mock_tess.Output = MagicMock()

    tokens = extract_tokens_with_bboxes("ignored.png")

    assert tokens[0]["bbox"] == [50, 50, 100, 150]


def test_preprocessing_returns_inverse_transform_for_box_mapping(tmp_path):
    from app.ocr_processor import _preprocess_page_image_with_transform

    image = _make_test_image(width=100, height=80)
    image_path = tmp_path / "deskewed.png"
    cv2.imwrite(str(image_path), image)
    deskew_transform = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -4.0]], dtype=np.float32)

    with patch("app.ocr_processor._detect_orientation_rotation", return_value=0), patch(
        "app.ocr_processor._deskew_with_transform",
        return_value=(image, deskew_transform),
    ):
        _, processed_to_source, _ = _preprocess_page_image_with_transform(image_path)

    expected = np.linalg.inv(np.vstack((deskew_transform, [0.0, 0.0, 1.0])))[:2]
    np.testing.assert_allclose(processed_to_source, expected)


def test_min_page_tokens_constant():
    from app.ocr_processor import MIN_PAGE_TOKENS
    assert MIN_PAGE_TOKENS == 3


def test_blank_page_confidence_constant():
    from app.ocr_processor import BLANK_PAGE_CONFIDENCE
    assert BLANK_PAGE_CONFIDENCE == 30.0
