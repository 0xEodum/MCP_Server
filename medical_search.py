"""
medical_search.py — поисковые функции для медицинской системы

Реализует трехэтапный поиск по медицинским документам:
1. Нормализация запроса через disease_registry
2. Получение обзорной информации из disease_overview
3. Извлечение конкретных разделов из disease_sections
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as rest

from medical_qdrant_api import MedicalQdrantStore, DISEASE_REGISTRY, DISEASE_OVERVIEW, DISEASE_SECTIONS
from medical_embedding_api import MedicalEmbedder, extract_icd10_codes
from medical_models import (
    MedicalSearchResult,
    MedicalOverviewResponse,
    MedicalSectionsResponse,
)


# -----------------------------
# Этап 1: Нормализация запроса
# -----------------------------

def normalize_medical_query(
    store: MedicalQdrantStore,
    embedder: MedicalEmbedder,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.6
) -> Dict[str, Any]:
    """Этап 1: Нормализация запроса пользователя.

    Ищет в disease_registry подходящие заболевания по запросу.
    Поддерживает поиск по названию, синонимам и кодам МКБ.
    """
    start_time = time.time()

    # Поиск по кодам МКБ
    icd_matches = extract_icd10_codes(query)
    results_by_icd = []

    if icd_matches:
        # Точный поиск по кодам МКБ
        icd_results = store.search_diseases_by_icd(icd_matches)

        for result in icd_results:
            if result.payload:
                disease_id = result.payload.get("disease_id") or result.id
                results_by_icd.append({
                    "disease_id": disease_id,
                    "canonical_name": result.payload.get("canonical_name"),
                    "icd10_codes": result.payload.get("icd10_codes", []),
                    "score": 1.0,
                    "match_type": "icd_code"
                })

    # Семантический поиск по эмбеддингам
    query_vector = embedder.encode_single(query)
    semantic_results = store.search_diseases_by_vector(
        query_vector,
        top_k=top_k,
        score_threshold=score_threshold
    )

    results_by_semantic = []
    for result in semantic_results:
        if result.payload:
            disease_id = result.payload.get("disease_id") or result.id
            results_by_semantic.append({
                "disease_id": disease_id,
                "canonical_name": result.payload.get("canonical_name"),
                "icd10_codes": result.payload.get("icd10_codes", []),
                "score": float(result.score),
                "match_type": "semantic"
            })

    # Объединение результатов (приоритет МКБ кодам)
    all_results = results_by_icd + results_by_semantic

    # Удаление дубликатов по disease_id
    seen_ids = set()
    unique_results = []
    for result in all_results:
        if result["disease_id"] not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result["disease_id"])

    # Сортировка по score
    unique_results.sort(key=lambda x: x["score"], reverse=True)

    took_ms = int((time.time() - start_time) * 1000)

    return {
        "found_diseases": unique_results[:top_k],
        "total_found": len(unique_results),
        "has_icd_matches": len(results_by_icd) > 0,
        "took_ms": took_ms
    }


# -----------------------------
# Этап 2: Обзорная информация
# -----------------------------

def get_disease_overview(
    store: MedicalQdrantStore,
    embedder: MedicalEmbedder,
    disease_ids: List[str],
    query: Optional[str] = None,
    top_k: int = 5
) -> MedicalOverviewResponse:
    """Этап 2: Получение обзорной информации по disease_ids.

    Возвращает краткую информацию о заболеваниях с доступными разделами.
    """
    start_time = time.time()

    if not disease_ids:
        return MedicalOverviewResponse(found_diseases=[], total_found=0)

    results = []

    if query:
        # Семантический поиск с фильтром по disease_ids
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
                    "score": float(result.score)
                })
    else:
        # Получение всех записей по disease_ids
        overview_results = store.get_disease_overview(disease_ids)

        for result in overview_results:
            if result.payload:
                results.append({
                    "disease_id": result.payload.get("disease_id"),
                    "canonical_name": result.payload.get("canonical_name"),
                    "icd10_primary": result.payload.get("icd10_primary"),
                    "summary": result.payload.get("summary", ""),
                    "available_sections": result.payload.get("available_sections", []),
                    "score": 1.0
                })

    took_ms = int((time.time() - start_time) * 1000)

    return MedicalOverviewResponse(
        found_diseases=results,
        total_found=len(results),
        took_ms=took_ms
    )


# -----------------------------
# Этап 3: Получение разделов
# -----------------------------

def get_disease_sections(
    store: MedicalQdrantStore,
    disease_id: str,
    section_ids: Optional[List[str]] = None,
    query: Optional[str] = None,
    embedder: Optional[MedicalEmbedder] = None,
    top_k: int = 10
) -> MedicalSectionsResponse:
    """Этап 3: Получение конкретных разделов документа.

    Args:
        disease_id: ID заболевания
        section_codes: коды разделов (например ["1.1", "2.3"])
        section_ids: ID разделов из исходного JSON
        query: дополнительный поисковый запрос
        embedder: для семантического поиска по query
        top_k: максимум результатов
    """
    start_time = time.time()

    sections = []
    canonical_name = ""

    if query and embedder:
        # Семантический поиск с фильтрами
        query_vector = embedder.encode_single(query)

        search_results = store.get_disease_sections(
            disease_id,
            section_ids=section_ids,
            query_vector=query_vector,
            top_k=top_k
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
                    "score": float(result.score)
                })
    else:
        # Получение по фильтрам
        section_results = store.get_disease_sections(
            disease_id,
            section_ids=section_ids,
            top_k=top_k
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
                    "score": 1.0
                })

    # Сортировка по section_id для последовательности
    sections.sort(key=lambda x: x.get("section_id", ""))

    took_ms = int((time.time() - start_time) * 1000)

    return MedicalSectionsResponse(
        disease_id=disease_id,
        canonical_name=canonical_name,
        sections=sections,
        total_sections=len(sections),
        took_ms=took_ms
    )


# -----------------------------
# Комплексный поиск
# -----------------------------

def medical_search_workflow(
    store: MedicalQdrantStore,
    embedder: MedicalEmbedder,
    user_query: str,
    max_diseases: int = 3,
    include_sections: bool = False,
    section_query: Optional[str] = None
) -> Dict[str, Any]:
    """Полный workflow медицинского поиска.

    Выполняет все 3 этапа последовательно для демонстрации.
    В реальности LLM должен вызывать этапы отдельно.
    """

    # Этап 1: Нормализация
    normalized = normalize_medical_query(store, embedder, user_query, top_k=max_diseases)

    if not normalized["found_diseases"]:
        return {
            "stage": "normalization",
            "result": "Не найдено подходящих заболеваний",
            "query": user_query
        }

    # Этап 2: Обзорная информация
    disease_ids = [d["disease_id"] for d in normalized["found_diseases"]]
    overview = get_disease_overview(store, embedder, disease_ids)

    result = {
        "stage": "overview",
        "normalization": normalized,
        "overview": overview,
        "query": user_query
    }

    # Этап 3: Получение разделов (опционально)
    if include_sections and overview.found_diseases:
        primary_disease_id = overview.found_diseases[0]["disease_id"]
        sections = get_disease_sections(
            store,
            primary_disease_id,
            query=section_query or user_query,
            embedder=embedder
        )
        result["sections"] = sections
        result["stage"] = "sections"

    return result


__all__ = [
    "normalize_medical_query",
    "get_disease_overview",
    "get_disease_sections",
    "medical_search_workflow",
    "extract_icd_codes",
]