"""Index disease sections into the dedicated Qdrant collection."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, TYPE_CHECKING

from ..constants import DISEASE_SECTIONS
from ..models import DiseaseSectionPayload, MedicalDocument
from ..utils import slugify_document_title

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def index_disease_sections(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    documents: List[MedicalDocument],
    *,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """Encode and persist content sections for every indexed disease."""

    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []
    ids: List[str] = []
    total_sections = 0

    for doc in documents:
        disease_id = slugify_document_title(doc.doc_title)

        for section in doc.sections:
            body = section.get('body', '').strip()
            title = section.get('title', '').strip()
            section_id = section.get('id', '')

            if not body or len(body) < 10:
                continue

            vector = embedder.encode_single(f"{title} {body}".strip())

            payload = DiseaseSectionPayload(
                disease_id=disease_id,
                canonical_name=doc.doc_title,
                section_id=section_id,
                section_title=title,
                content=body,
                content_length=len(body),
            )

            vectors.append(vector)
            payloads.append(asdict(payload))
            ids.append(f"{disease_id}#{section_id}")
            total_sections += 1

    if vectors:
        for start in range(0, len(vectors), batch_size):
            store.upsert_to_collection(
                DISEASE_SECTIONS,
                vectors=vectors[start:start + batch_size],
                payloads=payloads[start:start + batch_size],
                ids=ids[start:start + batch_size],
            )

    return {
        "collection": DISEASE_SECTIONS,
        "indexed": total_sections,
    }


__all__ = ["index_disease_sections"]
