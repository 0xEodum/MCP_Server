"""Fetch disease overview records for selected identifiers."""
from __future__ import annotations

import time
from typing import List, Optional, TYPE_CHECKING

from ..models import MedicalOverviewResponse

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def get_disease_overview(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    disease_ids: List[str],
    query: Optional[str] = None,
    top_k: int = 5,
) -> MedicalOverviewResponse:
    """Return overview payloads for one or more diseases, optionally reranked by query."""

    start_time = time.time()

    if not disease_ids:
        return MedicalOverviewResponse(found_diseases=[], total_found=0)

    results = []

    if query:
        query_vector = embedder.encode_single(query)
        search_results = store.get_disease_overview(disease_ids, query_vector, top_k)

        for result in search_results:
            if result.payload:
                results.append({
                    "disease_id": result.payload.get("disease_id"),
                    "canonical_name": result.payload.get("canonical_name"),
                    "icd10_primary": result.payload.get("icd10_primary"),
                    "summary": result.payload.get("summary", ""),
                    "available_sections": result.payload.get("available_sections", []),
                    "score": float(result.score),
                })
    else:
        overview_results = store.get_disease_overview(disease_ids)

        for result in overview_results:
            if result.payload:
                results.append({
                    "disease_id": result.payload.get("disease_id"),
                    "canonical_name": result.payload.get("canonical_name"),
                    "icd10_primary": result.payload.get("icd10_primary"),
                    "summary": result.payload.get("summary", ""),
                    "available_sections": result.payload.get("available_sections", []),
                    "score": 1.0,
                })

    took_ms = int((time.time() - start_time) * 1000)

    return MedicalOverviewResponse(
        found_diseases=results,
        total_found=len(results),
        took_ms=took_ms,
    )


__all__ = ["get_disease_overview"]
