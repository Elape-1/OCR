from app.routing_engine import build_attribute_payloads, cluster_entity_tokens


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
