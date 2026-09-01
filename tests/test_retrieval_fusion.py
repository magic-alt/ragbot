from benchmarks.pdf_scale import _document_marker
from services.api.app.retrieval.rerank import rrf_fuse


def test_rrf_keeps_disjoint_secondary_candidates_in_final_window():
    primary = [(f"vector-{index}", 1.0 - index / 100.0) for index in range(10)]
    secondary = [(f"lexical-{index}", 1.0 - index / 100.0) for index in range(10)]

    fused = rrf_fuse(primary, secondary)
    top_five = [item_id for item_id, _ in fused[:5]]

    assert "lexical-0" in top_five
    assert any(item_id.startswith("vector-") for item_id in top_five)
    assert any(item_id.startswith("lexical-") for item_id in top_five)


def test_pdf_scale_markers_have_four_unique_terms_per_document():
    markers = [_document_marker(index).split() for index in range(1000)]

    assert all(len(parts) == 4 for parts in markers)
    flattened = [term for parts in markers for term in parts]
    assert len(flattened) == 4000
    assert len(set(flattened)) == 4000
