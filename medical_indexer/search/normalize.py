"""Normalize user queries into ranked disease candidates."""
from __future__ import annotations

import time
from typing import Any, Dict, List, TYPE_CHECKING

from ..embeddings import extract_icd10_codes

if TYPE_CHECKING:
    from ..embeddings import MedicalEmbedder
    from ..qdrant import MedicalQdrantStore


def normalize_medical_query(
    store: 'MedicalQdrantStore',
    embedder: 'MedicalEmbedder',
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.6,
    enable_reranking: bool = True,
    rerank_top_k: int = 20,
) -> Dict[str, Any]:
    """Return candidate diseases matching the query by ICD codes and semantic search."""

    start_time = time.time()

    icd_matches = extract_icd10_codes(query)
    results_by_icd: List[Dict[str, Any]] = []

    if icd_matches:
        icd_results = store.search_diseases_by_icd(icd_matches)
        for result in icd_results:
            if result.payload:
                disease_id = result.payload.get("disease_id") or result.id
                results_by_icd.append({
                    "disease_id": disease_id,
                    "canonical_name": result.payload.get("canonical_name"),
                    "icd10_codes": result.payload.get("icd10_codes", []),
                    "score": 1.0,
                    "match_type": "icd_code",
                })

    query_vector = embedder.encode_single(query)
    search_limit = rerank_top_k if enable_reranking else top_k

    semantic_results = store.search_diseases_by_vector(
        query_vector,
        top_k=search_limit,
        score_threshold=score_threshold,
    )

    results_by_semantic: List[Dict[str, Any]] = []
    for result in semantic_results:
        if result.payload:
            disease_id = result.payload.get("disease_id") or result.id
            results_by_semantic.append({
                "disease_id": disease_id,
                "canonical_name": result.payload.get("canonical_name"),
                "icd10_codes": result.payload.get("icd10_codes", []),
                "score": float(result.score),
                "match_type": "semantic",
            })

    if enable_reranking and results_by_semantic:
        reranked_results = embedder.rerank_results(
            query=query,
            candidates=results_by_semantic,
            text_field="canonical_name",
            top_k=top_k,
        )
        results_by_semantic = reranked_results

    all_results = results_by_icd + results_by_semantic

    seen_ids = set()
    unique_results: List[Dict[str, Any]] = []
    for result in all_results:
        if result["disease_id"] not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result["disease_id"])

    unique_results.sort(key=lambda item: item["score"], reverse=True)

    took_ms = int((time.time() - start_time) * 1000)

    return {
        "found_diseases": unique_results[:top_k],
        "total_found": len(unique_results),
        "has_icd_matches": bool(results_by_icd),
        "reranking_applied": enable_reranking and bool(results_by_semantic),
        "search_candidates": len(results_by_semantic) if enable_reranking else 0,
        "took_ms": took_ms,
    }


__all__ = ["normalize_medical_query"]
