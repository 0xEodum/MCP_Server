"""Index builders for the disease overview collection."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, TYPE_CHECKING

from ..constants import DISEASE_OVERVIEW
from ..models import DiseaseOverviewPayload, MedicalDocument
from ..utils import slugify_document_title

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def index_disease_overview(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    documents: List[MedicalDocument],
) -> Dict[str, Any]:
    """Create summary records with available sections and the leading ICD code."""

    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []
    ids: List[str] = []

    for doc in documents:
        disease_id = slugify_document_title(doc.doc_title)

        summary_parts: List[str] = []
        for section in doc.sections[:3]:
            body = section.get('body', '').strip()
            if body and len(body) > 10:
                summary_parts.append(body[:200])

        summary = (" ".join(summary_parts)[:500] + "...") if summary_parts else ""

        available_sections = []
        for section in doc.sections:
            if section.get('body') and section.get('title'):
                available_sections.append({
                    "id": section['id'],
                    "title": section['title'],
                    "has_content": len(section['body'].strip()) > 10,
                })

        embed_text = f"{doc.doc_title} {summary}".strip()
        vector = embedder.encode_single(embed_text)

        payload = DiseaseOverviewPayload(
            disease_id=disease_id,
            canonical_name=doc.doc_title,
            icd10_primary=doc.mkb[0] if doc.mkb else None,
            summary=summary,
            available_sections=available_sections,
        )

        vectors.append(vector)
        payloads.append(asdict(payload))
        ids.append(f"{disease_id}_overview")

    if vectors:
        store.upsert_to_collection(
            DISEASE_OVERVIEW,
            vectors=vectors,
            payloads=payloads,
            ids=ids,
        )

    return {
        "collection": DISEASE_OVERVIEW,
        "indexed": len(vectors),
    }


__all__ = ["index_disease_overview"]
