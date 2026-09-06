import os
import cv2
import numpy as np
from backend.app.preprocessing import DocumentPreprocessor


def make_text_image(text: str, size=(800, 1100), font_scale=3, thickness=5, bg=255, color=0):
    img = np.full((size[1], size[0]), bg, dtype=np.uint8)
    org = (50, size[1] // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return img


def test_clean_image_processing(tmp_path):
    pre = DocumentPreprocessor()
    img = make_text_image("TEST OCR")
    in_path = tmp_path / "clean.png"
    out_path = tmp_path / "clean_out.png"
    cv2.imwrite(str(in_path), img)

    res = pre.process(str(in_path), str(out_path))
    assert res["processed_image"] is not None
    assert os.path.exists(str(out_path))


def test_deskew_corrects_rotation():
    pre = DocumentPreprocessor()
    img = make_text_image("SKEW TEST")
    # rotate by 15 degrees
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    corrected, angle = pre.deskew(rot)
    assert abs(abs(angle) - 15) < 1.5


def test_clahe_improves_contrast():
    pre = DocumentPreprocessor()
    # low-contrast: background 120, text 130
    img = np.full((600, 800), 120, dtype=np.uint8)
    cv2.putText(img, "low", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, 130, 20, cv2.LINE_AA)

    before_std = float(np.std(img))
    after = pre.apply_clahe(img)
    after_std = float(np.std(after))
    assert after_std > before_std


def test_process_handles_missing_file_gracefully():
    pre = DocumentPreprocessor()
    res = pre.process("/nonexistent/path/does_not_exist.png")
    assert res["processed_image"] is None
