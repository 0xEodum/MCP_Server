from __future__ import annotations

import os
import time
import json
import threading
import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

# --- Medical RAG stack ---
from medical_indexer import (
    DISEASE_OVERVIEW,
    DISEASE_REGISTRY,
    DISEASE_SECTIONS,
    MedicalEmbedder,
    MedicalQdrantStore,
    get_disease_overview,
    get_disease_sections,
    index_medical_documents,
    medical_search_workflow,
    normalize_medical_query,
)

# --- Lab Analysis ---
from search_by_patterns.disease_search_engine import MedicalLabAnalyzer
from pymongo import MongoClient

from search_by_patterns.sync_manager import SyncManager

SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "3600"))

# --- MCP SDK ---
MCP_MODE = None
try:
    from mcp.server.fastmcp import FastMCP

    MCP_MODE = "fast"
except Exception:
    try:
        from mcp.server import Server, stdio

        MCP_MODE = "base"
    except Exception:
        MCP_MODE = None

# --------------------
# Configuration
# --------------------
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "medical_lab")

_medical_store: Optional[MedicalQdrantStore] = None
_medical_embedder: Optional[MedicalEmbedder] = None
_lab_analyzer: Optional[MedicalLabAnalyzer] = None
_mongodb_client: Optional[MongoClient] = None
_sync_manager: Optional[SyncManager] = None
_sync_thread: Optional[threading.Thread] = None
_sync_loop: Optional[asyncio.AbstractEventLoop] = None


def _run_sync_manager_in_thread(sync_manager: SyncManager):
    global _sync_loop

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _sync_loop = loop

    try:
        loop.run_until_complete(sync_manager.start())
    except Exception as e:
        print(f"❌ Sync manager error: {e}")
    finally:
        loop.close()


def _ensure_medical_deps() -> tuple[MedicalQdrantStore, MedicalEmbedder]:
    global _medical_store, _medical_embedder
    if _medical_store is None:
        _medical_store = MedicalQdrantStore(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if _medical_embedder is None:
        _medical_embedder = MedicalEmbedder(DEFAULT_MODEL)
    return _medical_store, _medical_embedder


def _ensure_lab_analyzer() -> MedicalLabAnalyzer:
    global _lab_analyzer, _mongodb_client, _sync_manager, _sync_thread

    if _lab_analyzer is None:
        if _mongodb_client is None:
            _mongodb_client = MongoClient(MONGODB_URI)
            _mongodb_client.admin.command('ping')
            print("✓ Connected to MongoDB for lab analysis")

        _lab_analyzer = MedicalLabAnalyzer(mongodb_client=_mongodb_client)
        _lab_analyzer.load_all_from_mongodb()
        print(f"✓ Lab analyzer initialized with MongoDB (db: {MONGODB_DB})")

        if _sync_manager is None and _sync_thread is None:
            print(f"🔄 Starting background sync (interval: {SYNC_INTERVAL}s)...")

            _sync_manager = SyncManager(
                analyzer=_lab_analyzer,
                mongodb_client=_mongodb_client,
                db_name=MONGODB_DB,
                check_interval=SYNC_INTERVAL
            )

            _sync_thread = threading.Thread(
                target=_run_sync_manager_in_thread,
                args=(_sync_manager,),
                daemon=True,
                name="SyncManagerThread"
            )
            _sync_thread.start()
            print("✓ Background sync thread started")

    return _lab_analyzer


def _pack(obj: Any) -> Any:
    """Convert dataclasses to JSON-compatible dicts."""
    if is_dataclass(obj):
        return {k: _pack(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_pack(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    return obj


async def t_ping() -> Dict[str, Any]:
    s, _ = _ensure_medical_deps()
    return {"ok": s.ping(), "ts": int(time.time())}


async def t_analyze_lab_tests(
        *,
        tests: List[Dict[str, str]],
        gender: str = "unisex",
        top_k: int = 10,
        categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    analyzer = _ensure_lab_analyzer()

    try:
        start_time = time.time()

        results = analyzer.analyze_patient(
            tests=tests,
            gender=gender,
            top_k=top_k,
            categories=categories
        )

        processing_time = (time.time() - start_time) * 1000

        disease_results = []
        for r in results:
            disease_results.append({
                "disease_id": r.disease_id,
                "canonical_name": r.canonical_name,
                "matched_patterns": r.matched_patterns,
                "total_patterns": r.total_patterns,
                "matched_score": r.matched_score,
                "contradiction_penalty": r.contradiction_penalty,
                "total_score": r.total_score,
                "max_possible_score": r.max_possible_score,
                "normalized_score": r.normalized_score,
                "matched_details": r.matched_details,
                "contradictions": r.contradictions,
                "missing_data": r.missing_data,
                "redundant_data": r.redundant_data,
                "expected_patterns": r.expected_patterns
            })

        return {
            "success": True,
            "processing_time_ms": processing_time,
            "results": disease_results,
            "total_found": len(disease_results),
            "tool": "analyze_lab_tests",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "analyze_lab_tests"
        }


async def t_explain_lab_tests(
        *,
        tests: List[Dict[str, str]],
        gender: str = "unisex"
) -> Dict[str, Any]:
    analyzer = _ensure_lab_analyzer()

    try:
        start_time = time.time()

        explanations = analyzer.explain_tests(
            tests=tests,
            gender=gender
        )

        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "processing_time_ms": processing_time,
            "explanations": explanations,
            "total_tests": len(explanations),
            "tool": "explain_lab_tests"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "explain_lab_tests"
        }


async def t_medical_normalize_query(
        *,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.6,
        enable_reranking: bool = True,
        rerank_top_k: int = 5,
) -> Dict[str, Any]:
    store, emb = _ensure_medical_deps()

    try:
        result = normalize_medical_query(
            store,
            emb,
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            enable_reranking=enable_reranking,
            rerank_top_k=rerank_top_k
        )
        raw_results = result.get("found_diseases", [])
        filtered_results = []

        for candidate in raw_results:
            score_value = candidate.get("score")

            if score_value is None:
                continue

            if score_value < 0:
                continue

            filtered_candidate = {
                key: value for key, value in candidate.items()
                if key not in {"rerank_score", "original_score"}
            }
            filtered_results.append((filtered_candidate, score_value))

        if filtered_results:
            max_score = max(score for _, score in filtered_results)
            normalized_results = []

            for filtered_candidate, raw_score in filtered_results:
                if max_score > 0:
                    filtered_candidate["score"] = raw_score / max_score
                else:
                    filtered_candidate["score"] = 0.0
                normalized_results.append(filtered_candidate)
        else:
            normalized_results = []

        result["found_diseases"] = normalized_results
        result["total_found"] = len(normalized_results)
        result["tool"] = "medical_normalize_query"
        result["stage"] = "normalization"
        return result
    except Exception as e:
        return {
            "tool": "medical_normalize_query",
            "error": str(e),
            "query": query
        }


async def t_medical_get_overview(
        *,
        disease_ids: List[str],
        query: Optional[str] = None,
        top_k: int = 5,
) -> Dict[str, Any]:
    store, emb = _ensure_medical_deps()

    try:
        result = get_disease_overview(
            store,
            emb,
            disease_ids,
            query=query,
            top_k=top_k
        )
        out = _pack(result)
        out["tool"] = "medical_get_overview"
        out["stage"] = "overview"
        return out
    except Exception as e:
        return {
            "tool": "medical_get_overview",
            "error": str(e),
            "disease_ids": disease_ids
        }


async def t_medical_get_sections(
        *,
        disease_id: str,
        section_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        top_k: int = 10,
) -> Dict[str, Any]:
    store, emb = _ensure_medical_deps()

    try:
        result = get_disease_sections(
            store,
            disease_id,
            section_ids=section_ids,
            query=query,
            embedder=emb if query else None,
            top_k=top_k
        )
        out = _pack(result)
        out["tool"] = "medical_get_sections"
        out["stage"] = "sections"
        return out
    except Exception as e:
        return {
            "tool": "medical_get_sections",
            "error": str(e),
            "disease_id": disease_id
        }



def _register_fast() -> None:
    """FastMCP registration with medical tools."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("medical-rag", host="0.0.0.0", port=8000)

    # Health check
    @mcp.tool()
    async def ping() -> dict:
        """Health check for Qdrant and embeddings."""
        return await t_ping()

    # Medical tools (primary)
    @mcp.tool()
    async def medical_normalize_query(
            query: str,
            top_k: int = 5,
            score_threshold: float = 0.6,
            enable_reranking: bool = True,
            rerank_top_k: int = 3,
    ) -> dict:
        """
        ЭТАП 1: Нормализация медицинского запроса

        Преобразует пользовательский запрос в список конкретных заболеваний.
        Это ПЕРВЫЙ инструмент, который нужно использовать для любого медицинского вопроса.

        КОГДА ИСПОЛЬЗОВАТЬ:
        - Пользователь спрашивает о симптомах, заболеваниях, лечении
        - Нужно найти конкретные болезни по описанию
        - Пользователь упомянул код МКБ-10
        - Начало медицинского поиска новой темы

        ПАРАМЕТРЫ:
        - query: нормализованный запрос пользователя, наименование болезни (например: "перикардиты", "ретинобластома")
        - top_k: сколько заболеваний найти (рекомендуется 3-5)
        - score_threshold: минимальная релевантность (0.6 подходит для большинства случаев)
        - enable_reranking: улучшает качество результатов (всегда используй True)
        - rerank_top_k: должно быть меньше или равно top_k

        ВОЗВРАЩАЕТ:
        - found_diseases: список найденных заболеваний с disease_id и названиями
        - has_icd_matches: найдены ли точные совпадения по МКБ-10
        - reranking_applied: применялся ли реранкинг для улучшения результатов

        СОВЕТ: После получения списка заболеваний используй medical_get_overview для получения детальной информации.
        ПРАВИЛО: НЕ ИСПОЛЬЗУЙ ДЛЯ ПОИСКА ИНФОРМАЦИИ ПО УЖЕ ИЗВЕСТНОМУ ДОКУМЕНТУ (например симптомы некоторой болезни, которую ты уже нашел ранее). Для такого поиска используй medical_get_sections
        """
        return await t_medical_normalize_query(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            enable_reranking=enable_reranking,
            rerank_top_k=rerank_top_k
        )

    @mcp.tool()
    async def medical_get_overview(
            disease_ids: List[str],
            query: Optional[str] = None,
            top_k: int = 5,
    ) -> dict:
        """
        ЭТАП 2: Получение обзорной информации о заболеваниях

        Возвращает краткое описание заболеваний и список доступных разделов документов.
        Используй ПОСЛЕ medical_normalize_query для получения оглавления.

        КОГДА ИСПОЛЬЗОВАТЬ:
        - Получил disease_ids от medical_normalize_query
        - Нужна общая информация о заболевании
        - Хочешь узнать, какие разделы доступны для изучения

        ПАРАМЕТРЫ:
        - disease_ids: список ID заболеваний (из результата medical_normalize_query)
        - query: дополнительный поисковый запрос для фильтрации (необязательно)
        - top_k: максимальное количество результатов

        ВОЗВРАЩАЕТ:
        - found_diseases: список с краткими описаниями заболеваний
        - available_sections: доступные разделы для каждого заболевания
            * id: идентификатор раздела для medical_get_sections
            * title: название раздела (например, "Симптомы", "Лечение")
            * has_content: есть ли содержимое в разделе

        СОВЕТ: Изучи available_sections и используй medical_get_sections для получения конкретной информации из интересующих разделов.
        """
        return await t_medical_get_overview(
            disease_ids=disease_ids,
            query=query,
            top_k=top_k
        )

    @mcp.tool()
    async def medical_get_sections(
            disease_id: str,
            section_ids: Optional[List[str]] = None,
            query: Optional[str] = None,
            top_k: int = 10,
    ) -> dict:
        """
        ЭТАП 3: Получение конкретных разделов медицинского документа
        Возвращает подробную информацию из конкретных разделов документа о заболевании.
        Это ФИНАЛЬНЫЙ этап для получения детальной медицинской информации.

        КОГДА ИСПОЛЬЗОВАТЬ:
        - Получил disease_id от medical_get_overview
        - Нужна детальная информация из конкретных разделов
        - Пользователь спрашивает о симптомах, лечении, диагностике конкретной болезни
        - Хочешь получить полный текст из медицинских руководств

        ПАРАМЕТРЫ:
        - disease_id: ID заболевания (из medical_get_overview)
        - section_ids: список ID разделов (из available_sections в overview)
            * Примеры: ["symptoms", "treatment", "diagnosis", "complications"]
            * Если не указать - получишь все доступные разделы
        - query: семантический поиск внутри разделов (например: "побочные эффекты")
        - top_k: максимальное количество разделов в ответе

        ВОЗВРАЩАЕТ:
        - disease_id: ID заболевания
        - canonical_name: официальное название заболевания
        - sections: список разделов с содержимым
            * section_id: ID раздела
            * section_title: название раздела
            * content: полный текст раздела
            * content_length: длина содержимого
            * score: релевантность (если использовался query)

        СТРАТЕГИИ ИСПОЛЬЗОВАНИЯ:
        Получить все разделы: medical_get_sections(disease_id="doc_abc123")
        Получить конкретные разделы: medical_get_sections(disease_id="doc_abc123", section_ids=["doc_terms", "doc_crat_info_1_1"])
        Семантический поиск внутри заболевания: medical_get_sections(disease_id="doc_abc123", query="анамнез")
        """
        return await t_medical_get_sections(
            disease_id=disease_id,
            section_ids=section_ids,
            query=query,
            top_k=top_k
        )

    @mcp.tool()
    async def analyze_lab_tests(
            tests: List[Dict[str, str]],
            gender: str = "unisex",
            top_k: int = 10,
            categories: Optional[List[str]] = None
    ) -> dict:
        """
        Анализ лабораторных тестов для определения возможных заболеваний

        Анализирует результаты лабораторных анализов с использованием улучшенной системы
        непрерывного скоринга и возвращает список возможных заболеваний с детальной оценкой.

        КОГДА ИСПОЛЬЗОВАТЬ:
        - У пациента есть результаты лабораторных анализов
        - Нужно определить возможные заболевания по отклонениям в анализах
        - Требуется дифференциальная диагностика на основе лабораторных данных

        ПАРАМЕТРЫ:
        - tests: список лабораторных тестов, каждый содержит:
            * name: название теста (например: "Гемоглобин", "Лейкоциты")
            * value: значение теста (например: "120", "8.5")
            * units: единицы измерения (например: "г/л", "×10^9/л")
        - gender: пол пациента ("male", "female", "unisex") для учета норм
        - top_k: максимальное количество заболеваний в результате (по умолчанию 10)
        - categories: фильтр по категориям заболеваний (необязательно)

        ВОЗВРАЩАЕТ:
        - success: успешность анализа
        - processing_time_ms: время обработки в миллисекундах
        - results: список найденных заболеваний с оценками:
            * disease_id: идентификатор заболевания
            * canonical_name: официальное название заболевания
            * matched_patterns: количество совпавших паттернов
            * total_patterns: общее количество паттернов для заболевания
            * matched_score: балл за совпадения (с учетом непрерывного скоринга)
            * contradiction_penalty: штраф за противоречия
            * total_score: итоговый балл
            * normalized_score: нормализованный балл (0-1)
            * matched_details: детали совпадений с contribution и gain
            * contradictions: список противоречий с reason и dist
            * missing_data: недостающие данные
            * redundant_data: нормальные значения вне паттерна (NEW в v2.0)
            * expected_patterns: полный список ожидаемых паттернов (NEW в v2.0)
        - total_found: общее количество найденных заболеваний
        """
        return await t_analyze_lab_tests(
            tests=tests,
            gender=gender,
            top_k=top_k,
            categories=categories
        )

    @mcp.tool()
    async def explain_lab_tests(
            tests: List[Dict[str, str]],
            gender: str = "unisex"
    ) -> dict:
        """
        Для каждого теста возвращает:
        - Каноническое название теста
        - Значение пациента (с конвертацией единиц)
        - Референсный диапазон для данного пола
        - Статус теста (normal, below_normal, above_normal, critically_low, critically_high)

        Это полезно для:
        - Понимания, какие тесты отклонились от нормы
        - Проверки конвертации единиц измерения
        - Валидации входных данных перед анализом
        - Объяснения пациенту статуса каждого теста

        ПАРАМЕТРЫ:
        - tests: список тестов [{"name": "...", "value": "...", "units": "..."}]
        - gender: пол пациента ("male", "female", "unisex")

        ВОЗВРАЩАЕТ:
        - success: успешность обработки
        - processing_time_ms: время обработки
        - explanations: список объяснений для каждого теста:
            * test_name: каноническое название и единицы
            * user_value: значение пациента (конвертированное)
            * reference_value: референсный диапазон (min, max)
            * status: статус теста

        """
        return await t_explain_lab_tests(
            tests=tests,
            gender=gender
        )

    mcp.run(transport="streamable-http")


def _register_base() -> None:
    """Base Server registration (stdio transport)."""
    # Similar registration but with base Server API
    # Skipping for brevity - would mirror the FastMCP version
    pass


if __name__ == "__main__":
    print("=== Medical RAG MCP Server v2.0 ===")
    print("Collections:")
    print(f"  - {DISEASE_REGISTRY} (disease registry)")
    print(f"  - {DISEASE_OVERVIEW} (disease overview)")
    print(f"  - {DISEASE_SECTIONS} (disease sections)")
    print()
    print("Lab Analysis Configuration:")
    print(f"  - MongoDB URI: {MONGODB_URI}")
    print(f"  - Database: {MONGODB_DB}")
    print(f"  - Sync interval: {SYNC_INTERVAL}s")
    print()
    print(f"Using model: {DEFAULT_MODEL}")
    print()

    if MCP_MODE == "fast":
        print("Starting with FastMCP...")
        _register_fast()
    elif MCP_MODE == "base":
        print("Starting with base MCP...")
        _register_base()
    else:
        print("MCP SDK not found.")
