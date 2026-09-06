from backend.app.exporter import _normalize_attr_boxes


def test_normalize_attr_boxes_fractional():
    boxes = [[0.1, 0.1, 0.2, 0.2]]
    out = _normalize_attr_boxes(boxes, page_width=1000, page_height=2000)
    assert isinstance(out[0][0], int)
    assert 0 <= out[0][0] <= 1000


def test_normalize_attr_boxes_pixels():
    boxes = [[100, 200, 300, 400]]
    out = _normalize_attr_boxes(boxes, page_width=1000, page_height=1000)
    assert out[0] == [100, 200, 300, 400] or all(0 <= v <= 1000 for v in out[0])


def test_normalize_attr_boxes_already_normalized():
    boxes = [[10, 20, 30, 40]]
    out = _normalize_attr_boxes(boxes, page_width=50, page_height=50)
    assert out[0] == [10, 20, 30, 40]


def test_normalize_attr_boxes_skips_malformed_values():
    boxes = [
        [10, 20, 30],
        ["left", 20, 30, 40],
        [float("nan"), 20, 30, 40],
        [10, 20, 30, 40],
    ]

    assert _normalize_attr_boxes(boxes, page_width=1000, page_height=1000) == [[10, 20, 30, 40]]
