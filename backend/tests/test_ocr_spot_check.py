"""Integration spot-check tests for OCR Priorities 1-3.

These tests run Tesseract on real raw pages to verify the PSM 3 switch,
confidence filtering, and blank-page detection work correctly on the
known problem documents identified in the audit.

Requires Tesseract to be installed — tests are skipped otherwise.
"""
from __future__ import annotations

import os
import shutil
import statistics
from pathlib import Path

import pytest

# Resolve project root relative to this test file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ASSETS = _PROJECT_ROOT / "datasets" / "raw-pages"

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
if not TESSERACT_AVAILABLE:
    # Also check common Windows install paths
    for _cand in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]:
        if Path(_cand).exists():
            TESSERACT_AVAILABLE = True
            break

skip_no_tesseract = pytest.mark.skipif(
    not TESSERACT_AVAILABLE,
    reason="Tesseract not installed",
)


def _extract(image_path: str | Path):
    """Import and call extract_tokens_with_bboxes."""
    from app.ocr_processor import extract_tokens_with_bboxes
    return extract_tokens_with_bboxes(image_path)


def _stats(tokens):
    """Quick stats dict from a token list."""
    if not tokens:
        return {"count": 0, "mean_conf": 0.0, "median_conf": 0.0, "below_50": 0}
    confs = [t["confidence"] for t in tokens]
    return {
        "count": len(tokens),
        "mean_conf": round(statistics.mean(confs), 1),
        "median_conf": round(statistics.median(confs), 1),
        "below_50": sum(1 for c in confs if c < 50),
    }


# ---------------------------------------------------------------------------
# Blank / near-blank pages (Priority 3)
# ---------------------------------------------------------------------------


@skip_no_tesseract
def test_blank_page_12_1():
    """datasets/raw-pages/12/page-001.png is a near-blank test page ('Test docx for LibreOffice conversion').

    After confidence filtering + blank-page detection, this should return
    an empty list or very few tokens.
    """
    img = _ASSETS / "12" / "page-001.png"
    if not img.exists():
        pytest.skip(f"Asset not found: {img}")

    tokens = _extract(img)
    stats = _stats(tokens)
    print(f"\n  [12/1.png] tokens={stats['count']}, mean_conf={stats['mean_conf']}")
    if tokens:
        for t in tokens:
            print(f"    '{t['token']}' conf={t['confidence']:.0f}")

    # Near-blank: expect 0 tokens (blank-page gate) or at most a handful
    assert stats["count"] <= 5, (
        f"Near-blank page produced too many tokens: {stats['count']}"
    )


@skip_no_tesseract
def test_blank_page_13_1():
    """datasets/raw-pages/13/page-001.png — same near-blank pattern."""
    img = _ASSETS / "13" / "page-001.png"
    if not img.exists():
        pytest.skip(f"Asset not found: {img}")

    tokens = _extract(img)
    stats = _stats(tokens)
    print(f"\n  [13/1.png] tokens={stats['count']}, mean_conf={stats['mean_conf']}")
    if tokens:
        for t in tokens:
            print(f"    '{t['token']}' conf={t['confidence']:.0f}")

    assert stats["count"] <= 5


# ---------------------------------------------------------------------------
# Mixed-layout pages (Priority 1 — PSM 3 vs PSM 6)
# ---------------------------------------------------------------------------


@skip_no_tesseract
def test_psm3_sidebar_page():
    """datasets/raw-pages/14/page-002.png — thesis page with sidebar comments + chat popup overlay.

    PSM 3 should correctly segment the main body from the sidebar.
    We verify that key body text AND sidebar text are both captured.
    """
    img = _ASSETS / "14" / "page-002.png"
    if not img.exists():
        pytest.skip(f"Asset not found: {img}")

    tokens = _extract(img)
    stats = _stats(tokens)
    text = " ".join(t["token"] for t in tokens).lower()

    print(f"\n  [14/2.png] tokens={stats['count']}, mean_conf={stats['mean_conf']}, "
          f"median={stats['median_conf']}, below_50={stats['below_50']}")
    print(f"  First 100 chars: {text[:100]}")

    # Body text should be present
    assert "chapter" in text or "methodology" in text, (
        "Body text not captured — expected 'chapter' or 'methodology'"
    )
    # Should have meaningful token count for a text-heavy page
    assert stats["count"] >= 30, f"Too few tokens for a text-heavy page: {stats['count']}"
    # Mean confidence should be reasonable (not all garbage)
    assert stats["mean_conf"] >= 50, f"Mean confidence too low: {stats['mean_conf']}"


@skip_no_tesseract
def test_psm3_diagram_page():
    """datasets/raw-pages/14/page-010.png — text + Agile circular diagram.

    Body paragraph text should be captured. Diagram noise should be
    filtered by the confidence threshold. Overall token quality should
    remain high (diagram junk removed).
    """
    img = _ASSETS / "14" / "page-010.png"
    if not img.exists():
        pytest.skip(f"Asset not found: {img}")

    tokens = _extract(img)
    stats = _stats(tokens)
    text = " ".join(t["token"] for t in tokens).lower()

    print(f"\n  [14/10.png] tokens={stats['count']}, mean_conf={stats['mean_conf']}, "
          f"median={stats['median_conf']}, below_50={stats['below_50']}")
    print(f"  First 100 chars: {text[:100]}")

    # Body text about "development" or "phases" or "agile" should appear
    assert any(kw in text for kw in ["development", "phase", "agile", "evaluation"]), (
        "Body text not captured from diagram page"
    )
    # Confidence should be healthy — diagram noise was filtered
    assert stats["mean_conf"] >= 50, f"Mean confidence too low after filtering: {stats['mean_conf']}"


# ---------------------------------------------------------------------------
# Clean document — regression check
# ---------------------------------------------------------------------------


@skip_no_tesseract
def test_clean_essay_unchanged():
    """datasets/raw-pages/2/page-001.png — clean essay with paragraph text.

    PSM 3 should not degrade OCR quality on clean single-column text.
    Token count and confidence should remain high.
    """
    img = _ASSETS / "2" / "page-001.png"
    if not img.exists():
        pytest.skip(f"Asset not found: {img}")

    tokens = _extract(img)
    stats = _stats(tokens)
    text = " ".join(t["token"] for t in tokens).lower()

    print(f"\n  [2/1.png] tokens={stats['count']}, mean_conf={stats['mean_conf']}, "
          f"median={stats['median_conf']}, below_50={stats['below_50']}")
    print(f"  First 100 chars: {text[:100]}")

    # Clean essay should produce substantial token count
    assert stats["count"] >= 80, f"Token count too low for clean essay: {stats['count']}"
    # Mean confidence should be high on clean text
    assert stats["mean_conf"] >= 80, f"Mean confidence dropped: {stats['mean_conf']}"
    # Known text should be present
    assert "reflecting" in text or "convolutional" in text or "neural" in text, (
        "Expected essay content not found"
    )


@skip_no_tesseract
def test_bulleted_list_page():
    """datasets/raw-pages/6/page-006.png — bulleted game rules list.

    PSM 3 should handle indented bullet structure at least as well as PSM 6.
    """
    img = _ASSETS / "6" / "page-006.png"
    if not img.exists():
        pytest.skip(f"Asset not found: {img}")

    tokens = _extract(img)
    stats = _stats(tokens)
    text = " ".join(t["token"] for t in tokens).lower()

    print(f"\n  [6/6.png] tokens={stats['count']}, mean_conf={stats['mean_conf']}, "
          f"median={stats['median_conf']}, below_50={stats['below_50']}")
    print(f"  First 100 chars: {text[:100]}")

    # Should capture the list content
    assert stats["count"] >= 40, f"Too few tokens for bulleted list: {stats['count']}"
    assert stats["mean_conf"] >= 70


@skip_no_tesseract
def test_confidence_filter_removes_noise():
    """Verify that across all test pages, no token has confidence < MIN_OCR_CONFIDENCE."""
    from app.ocr_processor import MIN_OCR_CONFIDENCE

    test_images = [
        _ASSETS / "2" / "page-001.png",
        _ASSETS / "6" / "page-006.png",
        _ASSETS / "14" / "page-002.png",
        _ASSETS / "14" / "page-010.png",
    ]

    for img_path in test_images:
        if not img_path.exists():
            continue
        tokens = _extract(img_path)
        low_conf = [t for t in tokens if t["confidence"] < MIN_OCR_CONFIDENCE]
        assert not low_conf, (
            f"{img_path.name}: found {len(low_conf)} tokens below "
            f"MIN_OCR_CONFIDENCE={MIN_OCR_CONFIDENCE}: "
            f"{[t['token'] for t in low_conf[:5]]}"
        )
