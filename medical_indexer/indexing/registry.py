"""Helpers for indexing disease registry data."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, TYPE_CHECKING

from ..constants import DISEASE_REGISTRY
from ..models import DiseaseRegistryPayload, MedicalDocument
from ..utils import slugify_document_title

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def index_disease_registry(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    documents: List[MedicalDocument],
) -> Dict[str, Any]:
    """Index canonical disease entries (title + ICD codes) into Qdrant."""

    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []
    disease_ids: List[str] = []

    for doc in documents:
        disease_id = slugify_document_title(doc.doc_title)
        vector = embedder.encode_single(doc.doc_title)

        payload = DiseaseRegistryPayload(
            canonical_name=doc.doc_title,
            icd10_codes=doc.mkb,
            disease_id=disease_id,
            canonical_name_lc=doc.doc_title.lower(),
        )

        vectors.append(vector)
        payloads.append(asdict(payload))
        disease_ids.append(disease_id)

    if vectors:
        store.upsert_to_collection(
            DISEASE_REGISTRY,
            vectors=vectors,
            payloads=payloads,
            ids=disease_ids,
        )

    return {
        "collection": DISEASE_REGISTRY,
        "indexed": len(vectors),
        "disease_ids": disease_ids,
    }


__all__ = ["index_disease_registry"]
