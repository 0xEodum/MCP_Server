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

async def t_medical_index_documents(
        *,
        json_files: List[str],
        recreate_collections: bool = False,
) -> Dict[str, Any]:
    """Индексация медицинских JSON документов в 3 коллекции."""
    store, emb = _ensure_medical_deps()

    # Проверка файлов
    valid_files = []
    for file_path in json_files:
        if not os.path.exists(file_path):
            continue
        valid_files.append(Path(file_path))

    if not valid_files:
        return {
            "tool": "medical_index_documents",
            "error": "Не найдено ни одного валидного файла",
            "provided_files": json_files
        }

    try:
        results = index_medical_documents(store, emb, valid_files, recreate_collections)
        results["tool"] = "medical_index_documents"
        return results
    except Exception as e:
        return {
            "tool": "medical_index_documents",
            "error": str(e),
            "files_attempted": [str(f) for f in valid_files]
        }


async def t_medical_normalize_query(
        *,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.6,
        enable_reranking: bool = True,
        rerank_top_k: int = 20,
) -> Dict[str, Any]:
    """ЭТАП 1: Нормализация медицинского запроса с реранкингом.

    Преобразует пользовательский запрос в конкретные заболевания.
    Поддерживает поиск по названию, синонимам и кодам МКБ.
    Включает реранкинг для повышения качества результатов.
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
    """ЭТАП 2: Получение обзорной информации о заболеваниях.

    Возвращает краткое описание и доступные разделы.
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
    """ЭТАП 3: Получение конкретных разделов документа.

    Args:
        disease_id: ID заболевания (обязательно)
        section_ids: ID разделов из JSON
        query: семантический поиск по содержимому
        top_k: максимум результатов
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


async def t_medical_search_workflow(
        *,
        user_query: str,
        max_diseases: int = 3,
        include_sections: bool = False,
        section_query: Optional[str] = None,
        enable_reranking: bool = True,
) -> Dict[str, Any]:
    """ДЕМО: Полный медицинский workflow за один вызов с реранкингом.

    В реальности LLM должен вызывать этапы поэтапно для проактивности.
    """
    store, emb = _ensure_medical_deps()

    try:
        result = medical_search_workflow(
            store,
            emb,
            user_query,
            max_diseases=max_diseases,
            include_sections=include_sections,
            section_query=section_query,
            enable_reranking=enable_reranking
        )
        result["tool"] = "medical_search_workflow"
        return result
    except Exception as e:
        return {
            "tool": "medical_search_workflow",
            "error": str(e),
            "query": user_query
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
    async def medical_index_documents(
            json_files: List[str],
            recreate_collections: bool = False,
    ) -> dict:
        """Index medical JSON documents into 3 Qdrant collections."""
        return await t_medical_index_documents(
            json_files=json_files,
            recreate_collections=recreate_collections
        )

    @mcp.tool()
    async def medical_normalize_query(
            query: str,
            top_k: int = 5,
            score_threshold: float = 0.6,
            enable_reranking: bool = True,
            rerank_top_k: int = 20,
    ) -> dict:
        """STAGE 1: Normalize user query to find specific diseases with reranking."""
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
        """STAGE 2: Get disease overview with available sections."""
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
        """STAGE 3: Get specific document sections by IDs or semantic search."""
        return await t_medical_get_sections(
            disease_id=disease_id,
            section_ids=section_ids,
            query=query,
            top_k=top_k
        )

    @mcp.tool()
    async def medical_search_workflow(
            user_query: str,
            max_diseases: int = 3,
            include_sections: bool = False,
            section_query: Optional[str] = None,
            enable_reranking: bool = True,
    ) -> dict:
        """DEMO: Complete medical search workflow with reranking (all stages)."""
        return await t_medical_search_workflow(
            user_query=user_query,
            max_diseases=max_diseases,
            include_sections=include_sections,
            section_query=section_query,
            enable_reranking=enable_reranking
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