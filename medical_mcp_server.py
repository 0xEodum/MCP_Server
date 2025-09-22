from __future__ import annotations

import os
import time
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

# --- Medical RAG stack ---
from medical_qdrant_api import MedicalQdrantStore
from medical_embedding_api import MedicalEmbedder
from medical_indexer import (
    index_medical_documents,
    DISEASE_REGISTRY,
    DISEASE_OVERVIEW,
    DISEASE_SECTIONS,
)
from medical_search import (
    normalize_medical_query,
    get_disease_overview,
    get_disease_sections,
    medical_search_workflow,
)

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

_medical_store: Optional[MedicalQdrantStore] = None
_medical_embedder: Optional[MedicalEmbedder] = None

def _ensure_medical_deps() -> tuple[MedicalQdrantStore, MedicalEmbedder]:
    """Инициализация медицинских компонентов."""
    global _medical_store, _medical_embedder
    if _medical_store is None:
        _medical_store = MedicalQdrantStore(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if _medical_embedder is None:
        _medical_embedder = MedicalEmbedder(DEFAULT_MODEL)
    return _medical_store, _medical_embedder



def _pack(obj: Any) -> Any:
    """Convert dataclasses to JSON-compatible dicts."""
    if is_dataclass(obj):
        return {k: _pack(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_pack(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    return obj


# --------------------
# Original RAG Tools
# --------------------
async def t_ping() -> Dict[str, Any]:
    s, _ = _ensure_medical_deps()
    return {"ok": s.ping(), "ts": int(time.time())}


# --------------------
# Medical Tools
# --------------------

async def t_medical_normalize_query(
        *,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.6,
        enable_reranking: bool = True,
        rerank_top_k: int = 5,
) -> Dict[str, Any]:
    """
    ЭТАП 1: Нормализация медицинского запроса
    """
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
    """
    ЭТАП 2: Получение обзорной информации о заболеваниях
    """
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
    """
    ЭТАП 3: Получение конкретных разделов медицинского документа
    """
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


# --------------------
# MCP Registration
# --------------------

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
        - Начало любого медицинского поиска

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

        💡 СОВЕТ: После получения списка заболеваний используй medical_get_overview для получения детальной информации.
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
        Используй ПОСЛЕ medical_normalize_query для получения детальной информации.

        КОГДА ИСПОЛЬЗОВАТЬ:
        - Получил disease_ids от medical_normalize_query
        - Нужна общая информация о заболевании
        - Хочешь узнать, какие разделы доступны для изучения
        - Нужно краткое описание перед углублением в детали

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
        Получить все разделы:
        medical_get_sections(disease_id="doc_abc123")
        Получить конкретные разделы:
        medical_get_sections(disease_id="doc_abc123", section_ids=["doc_terms", "doc_crat_info_1_1"])
        Семантический поиск внутри заболевания:
        medical_get_sections(disease_id="doc_abc123", query="анамнез")
        """
        return await t_medical_get_sections(
            disease_id=disease_id,
            section_ids=section_ids,
            query=query,
            top_k=top_k
        )

    mcp.run(transport="streamable-http")


def _register_base() -> None:
    """Base Server registration (stdio transport)."""
    # Similar registration but with base Server API
    # Skipping for brevity - would mirror the FastMCP version
    pass


if __name__ == "__main__":
    print("=== Medical RAG MCP Server ===")
    print("Collections:")
    print(f"  - {DISEASE_REGISTRY} (disease registry)")
    print(f"  - {DISEASE_OVERVIEW} (disease overview)")
    print(f"  - {DISEASE_SECTIONS} (disease sections)")
    print()
    print("Medical workflow:")
    print("  1. medical_normalize_query - find diseases by user query (with reranking)")
    print("  2. medical_get_overview - get disease info + available sections")
    print("  3. medical_get_sections - get specific sections content")
    print()
    print(f"Using model: {DEFAULT_MODEL}")
    print(f"Vector size: Expected ~1024 (E5-Large)")
    print()

    if MCP_MODE == "fast":
        print("Starting with FastMCP...")
        _register_fast()
    elif MCP_MODE == "base":
        print("Starting with base MCP...")
        _register_base()
    else:
        print("MCP SDK not found. Install: pip install mcp")
        print("Then run: python medical_mcp_server.py")