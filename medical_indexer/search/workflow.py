"""End-to-end search workflow orchestrating all search stages."""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from .normalize import normalize_medical_query
from .overview import get_disease_overview
from .sections import get_disease_sections

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def medical_search_workflow(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    user_query: str,
    max_diseases: int = 3,
    include_sections: bool = False,
    section_query: Optional[str] = None,
    enable_reranking: bool = True,
) -> Dict[str, Any]:
    """Run normalization, overview fetch, and optional section retrieval."""

    normalized = normalize_medical_query(
        store,
        embedder,
        user_query,
        top_k=max_diseases,
        enable_reranking=enable_reranking,
    )

    if not normalized["found_diseases"]:
        return {
            "stage": "normalization",
            "result": "No diseases matched the supplied query.",
            "query": user_query,
        }

    disease_ids = [d["disease_id"] for d in normalized["found_diseases"]]
    overview = get_disease_overview(store, embedder, disease_ids)

    result: Dict[str, Any] = {
        "stage": "overview",
        "normalization": normalized,
        "overview": overview,
        "query": user_query,
    }

    if include_sections and overview.found_diseases:
        primary_disease_id = overview.found_diseases[0]["disease_id"]
        sections = get_disease_sections(
            store,
            primary_disease_id,
            query=section_query or user_query,
            embedder=embedder,
        )
        result["sections"] = sections
        result["stage"] = "sections"

    return result


__all__ = ["medical_search_workflow"]
