"""High-level orchestration for indexing medical JSON documents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TYPE_CHECKING

from ..models import MedicalDocument
from .overview import index_disease_overview
from .registry import index_disease_registry
from .sections import index_disease_sections

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def _load_documents(json_files: Iterable[str | Path]) -> List[MedicalDocument]:
    documents: List[MedicalDocument] = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            documents.append(
                MedicalDocument(
                    doc_title=data['doc_title'],
                    mkb=data.get('mkb', []),
                    chapters=data.get('chapters', []),
                    sections=data.get('sections', []),
                )
            )
        except Exception as exc:  # pragma: no cover - logged for operator awareness
            print(f"Warning: unable to read {file_path}: {exc}")
    return documents


def index_medical_documents(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    json_files: Sequence[str | Path],
    *,
    recreate_collections: bool = False,
    section_batch_size: int = 100,
) -> Dict[str, Any]:
    """Pipeline that ingests JSON files and populates all medical collections."""

    documents = _load_documents(json_files)
    if not documents:
        return {"error": "No valid documents detected in the provided sources."}

    store.ensure_medical_collections(embedder.get_vector_size(), recreate=recreate_collections)
    store.create_medical_indexes()

    results: Dict[str, Any] = {}

    registry_result = index_disease_registry(store, embedder, documents)
    results['registry'] = registry_result

    overview_result = index_disease_overview(store, embedder, documents)
    results['overview'] = overview_result

    sections_result = index_disease_sections(
        store,
        embedder,
        documents,
        batch_size=section_batch_size,
    )
    results['sections'] = sections_result

    results['summary'] = {
        "total_documents": len(documents),
        "collections_created": 3,
        "total_vectors": (
            registry_result['indexed'] +
            overview_result['indexed'] +
            sections_result['indexed']
        ),
    }

    return results


__all__ = ["index_medical_documents"]
