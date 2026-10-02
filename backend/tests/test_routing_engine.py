from app.routing_engine import build_attribute_payloads, build_ocr_baseline_payloads, cluster_entity_tokens


def test_build_attribute_payloads_uses_default_threshold_of_0_75() -> None:
    payloads = build_attribute_payloads(
        page_id=1,
        document_id=2,
        token_predictions=[
            {
                "token": "Invoice",
                "label": "ENTITY",
                "confidence": 0.8,
                "bbox": [0, 0, 10, 10],
            }
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["validation_status"] == "APPROVED"


def test_cluster_entity_tokens_merges_same_label_tokens_across_non_adjacent_input_order() -> None:
    clusters = cluster_entity_tokens(
        [
            {
                "token": "Acme",
                "label": "vendor_name",
                "confidence": 0.96,
                "bbox": [10, 10, 40, 20],
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 1,
            },
            {
                "token": "Invoice",
                "label": "document_title",
                "confidence": 0.91,
                "bbox": [200, 10, 260, 20],
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 1,
            },
            {
                "token": "Corp",
                "label": "vendor_name",
                "confidence": 0.94,
                "bbox": [10, 28, 42, 38],
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 2,
            },
        ]
    )

    assert len(clusters) == 2
    vendor_cluster = next(cluster for cluster in clusters if cluster.entity_type_label == "vendor_name")
    assert vendor_cluster.extracted_value == "Acme Corp"
    assert len(vendor_cluster.bounding_boxes) == 2


def test_cluster_entity_tokens_keeps_close_same_label_fields_separate_on_same_line() -> None:
    clusters = cluster_entity_tokens(
        [
            {
                "token": "Alpha",
                "label": "form_value",
                "confidence": 0.97,
                "bbox": [10, 10, 22, 20],
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 1,
            },
            {
                "token": "Beta",
                "label": "form_value",
                "confidence": 0.96,
                "bbox": [41, 10, 53, 20],
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 1,
            },
        ]
    )

    assert len(clusters) == 2
    assert [cluster.extracted_value for cluster in clusters] == ["Alpha", "Beta"]
    assert all(len(cluster.bounding_boxes) == 1 for cluster in clusters)


def test_cluster_entity_tokens_preserves_paragraph_boundaries() -> None:
    clusters = cluster_entity_tokens(
        [
            {
                "token": "First",
                "label": "long_text",
                "confidence": 0.96,
                "bbox": [10, 10, 40, 20],
                "block_num": 1,
                "paragraph_num": 1,
                "line_num": 1,
            },
            {
                "token": "paragraph",
                "label": "long_text",
                "confidence": 0.96,
                "bbox": [10, 30, 70, 40],
                "block_num": 1,
                "paragraph_num": 2,
                "line_num": 1,
            },
        ]
    )

    assert len(clusters) == 1
    assert clusters[0].extracted_value == "First\n\nparagraph"


def test_ocr_baseline_extracts_labeled_multiline_value() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="invoice",
        tokens=[
            {"token": "Bill", "bbox": [10, 10, 30, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "to:", "bbox": [32, 10, 50, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Acme", "bbox": [55, 10, 80, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Corporation", "bbox": [10, 30, 70, 40], "confidence": 90, "block_num": 1, "paragraph_num": 2, "line_num": 1},
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "bill_to"
    assert payloads[0]["extracted_value"] == "Acme\nCorporation"


def test_ocr_baseline_keeps_text_for_unknown_document_type() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="unknown",
        tokens=[
            {"token": "Unclassified", "bbox": [10, 10, 70, 20], "confidence": 95},
            {"token": "document", "bbox": [75, 10, 125, 20], "confidence": 90},
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "ocr_text"
    assert payloads[0]["extracted_value"] == "Unclassified document"


def test_ocr_baseline_keeps_text_when_known_schema_has_no_matching_labels() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="invoice",
        tokens=[
            {"token": "Letter", "bbox": [10, 10, 50, 20], "confidence": 95},
            {"token": "content", "bbox": [55, 10, 100, 20], "confidence": 90},
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "ocr_text"
    assert payloads[0]["extracted_value"] == "Letter content"


def test_ocr_baseline_splits_long_unstructured_text() -> None:
    tokens = [
        {"token": f"word{index}", "bbox": [index, 10, index + 5, 20], "confidence": 95, "block_num": 1, "paragraph_num": index // 20 + 1, "line_num": index}
        for index in range(160)
    ]

    payloads = build_ocr_baseline_payloads(1, 2, tokens, document_type="unknown")

    assert len(payloads) > 1
    assert all(len(item["extracted_value"]) <= 600 for item in payloads)
    assert payloads[0]["entity_type_label"] == "ocr_text_1"


def test_ocr_baseline_extracts_letter_fields() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="letter",
        tokens=[
            {"token": "To:", "bbox": [10, 10, 25, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Dr.", "bbox": [30, 10, 45, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Riah", "bbox": [50, 10, 80, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Subject:", "bbox": [10, 30, 55, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
            {"token": "Permission", "bbox": [60, 30, 120, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
        ],
    )

    assert [(item["entity_type_label"], item["extracted_value"]) for item in payloads] == [
        ("recipient", "Dr. Riah"),
        ("subject", "Permission"),
    ]


def test_ocr_baseline_keeps_letter_body_paragraphs() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="letter",
        tokens=[
            {"token": "To:", "bbox": [10, 10, 25, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Recipient", "bbox": [30, 10, 80, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Subject:", "bbox": [10, 30, 55, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
            {"token": "Request", "bbox": [60, 30, 110, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
            {"token": "This", "bbox": [10, 60, 35, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "is", "bbox": [40, 60, 50, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "the", "bbox": [55, 60, 75, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "body.", "bbox": [80, 60, 115, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "More", "bbox": [10, 80, 35, 90], "confidence": 95, "block_num": 2, "paragraph_num": 2, "line_num": 1},
            {"token": "body.", "bbox": [40, 80, 75, 90], "confidence": 95, "block_num": 2, "paragraph_num": 2, "line_num": 1},
        ],
    )

    assert [(item["entity_type_label"], item["extracted_value"]) for item in payloads] == [
        ("recipient", "Recipient"),
        ("subject", "Request"),
        ("body_3", "This is the body."),
        ("body_4", "More body."),
    ]
