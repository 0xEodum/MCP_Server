"""Retrieve detailed section content for a selected disease."""
from __future__ import annotations

import time
from typing import List, Optional, TYPE_CHECKING

from ..models import MedicalSectionsResponse

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def get_disease_sections(
    store: 'MedicalQdrantStore',
    disease_id: str,
    section_ids: Optional[List[str]] = None,
    query: Optional[str] = None,
    embedder: Optional['MedicalEmbedder'] = None,
    top_k: int = 10,
) -> MedicalSectionsResponse:
    """Return enriched section payloads, optionally scored by a user query."""

    start_time = time.time()

    sections = []
    canonical_name = ""

    if query and embedder:
        query_vector = embedder.encode_single(query)

        search_results = store.get_disease_sections(
            disease_id,
            section_ids=section_ids,
            query_vector=query_vector,
            top_k=top_k,
        )

        for result in search_results:
            if result.payload:
                if not canonical_name:
                    canonical_name = result.payload.get("canonical_name", "")

                sections.append({
                    "section_id": result.payload.get("section_id"),
                    "section_title": result.payload.get("section_title"),
                    "content": result.payload.get("content"),
                    "content_length": result.payload.get("content_length", 0),
                    "score": float(result.score),
                })
    else:
        section_results = store.get_disease_sections(
            disease_id,
            section_ids=section_ids,
            top_k=top_k,
        )

        for result in section_results:
            if result.payload:
                if not canonical_name:
                    canonical_name = result.payload.get("canonical_name", "")

                sections.append({
                    "section_id": result.payload.get("section_id"),
                    "section_title": result.payload.get("section_title"),
                    "content": result.payload.get("content"),
                    "content_length": result.payload.get("content_length", 0),
                    "score": 1.0,
                })

    sections.sort(key=lambda item: item.get("section_id", ""))

    took_ms = int((time.time() - start_time) * 1000)

    return MedicalSectionsResponse(
        disease_id=disease_id,
        canonical_name=canonical_name,
        sections=sections,
        total_sections=len(sections),
        took_ms=took_ms,
    )


__all__ = ["get_disease_sections"]
