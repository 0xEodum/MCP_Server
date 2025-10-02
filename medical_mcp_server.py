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
    """
    Р—Р°РїСѓСЃРєР°РµС‚ SyncManager РІ РѕС‚РґРµР»СЊРЅРѕРј РїРѕС‚РѕРєРµ СЃ СЃРѕР±СЃС‚РІРµРЅРЅС‹Рј event loop.
    Р­С‚Р° С„СѓРЅРєС†РёСЏ Р±СѓРґРµС‚ СЂР°Р±РѕС‚Р°С‚СЊ РІ С„РѕРЅРѕРІРѕРј СЂРµР¶РёРјРµ.
    """
    global _sync_loop

    # РЎРѕР·РґР°С‘Рј РЅРѕРІС‹Р№ event loop РґР»СЏ СЌС‚РѕРіРѕ РїРѕС‚РѕРєР°
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _sync_loop = loop

    try:
        # Р—Р°РїСѓСЃРєР°РµРј async РјРµС‚РѕРґ start() РІ СЌС‚РѕРј event loop
        loop.run_until_complete(sync_manager.start())
    except Exception as e:
        print(f"вќЊ Sync manager error: {e}")
    finally:
        loop.close()


def _ensure_medical_deps() -> tuple[MedicalQdrantStore, MedicalEmbedder]:
    """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РјРµРґРёС†РёРЅСЃРєРёС… РєРѕРјРїРѕРЅРµРЅС‚РѕРІ."""
    global _medical_store, _medical_embedder
    if _medical_store is None:
        _medical_store = MedicalQdrantStore(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if _medical_embedder is None:
        _medical_embedder = MedicalEmbedder(DEFAULT_MODEL)
    return _medical_store, _medical_embedder


def _ensure_lab_analyzer() -> MedicalLabAnalyzer:
    """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Р°РЅР°Р»РёР·Р°С‚РѕСЂР° Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… С‚РµСЃС‚РѕРІ СЃ С„РѕРЅРѕРІРѕР№ СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёРµР№."""
    global _lab_analyzer, _mongodb_client, _sync_manager, _sync_thread

    if _lab_analyzer is None:
        if _mongodb_client is None:
            _mongodb_client = MongoClient(MONGODB_URI)
            # РџСЂРѕРІРµСЂРєР° РїРѕРґРєР»СЋС‡РµРЅРёСЏ
            _mongodb_client.admin.command('ping')
            print("вњ“ Connected to MongoDB for lab analysis")

        # РЎРѕР·РґР°С‘Рј Р°РЅР°Р»РёР·Р°С‚РѕСЂ СЃ MongoDB РєР»РёРµРЅС‚РѕРј
        _lab_analyzer = MedicalLabAnalyzer(mongodb_client=_mongodb_client)
        _lab_analyzer.load_all_from_mongodb()
        print(f"вњ“ Lab analyzer initialized with MongoDB (db: {MONGODB_DB})")

        # Р—Р°РїСѓСЃРєР°РµРј С„РѕРЅРѕРІСѓСЋ СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёСЋ
        if _sync_manager is None and _sync_thread is None:
            print(f"рџ”„ Starting background sync (interval: {SYNC_INTERVAL}s)...")

            _sync_manager = SyncManager(
                analyzer=_lab_analyzer,
                mongodb_client=_mongodb_client,
                db_name=MONGODB_DB,
                check_interval=SYNC_INTERVAL
            )

            # Р—Р°РїСѓСЃРєР°РµРј sync manager РІ РѕС‚РґРµР»СЊРЅРѕРј daemon РїРѕС‚РѕРєРµ
            _sync_thread = threading.Thread(
                target=_run_sync_manager_in_thread,
                args=(_sync_manager,),
                daemon=True,  # daemon=True РѕР·РЅР°С‡Р°РµС‚, С‡С‚Рѕ РїРѕС‚РѕРє Р·Р°РІРµСЂС€РёС‚СЃСЏ РїСЂРё РІС‹С…РѕРґРµ РёР· РїСЂРѕРіСЂР°РјРјС‹
                name="SyncManagerThread"
            )
            _sync_thread.start()
            print("вњ“ Background sync thread started")

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


# --------------------
# Original RAG Tools
# --------------------
async def t_ping() -> Dict[str, Any]:
    s, _ = _ensure_medical_deps()
    return {"ok": s.ping(), "ts": int(time.time())}


# --------------------
# Lab Analysis Tools
# --------------------

async def t_analyze_lab_tests(
        *,
        tests: List[Dict[str, str]],
        gender: str = "unisex",
        top_k: int = 10,
        categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    РђРЅР°Р»РёР· Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… С‚РµСЃС‚РѕРІ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ РІРѕР·РјРѕР¶РЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№
    """
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
                "missing_data": r.missing_data
            })

        return {
            "success": True,
            "processing_time_ms": processing_time,
            "results": disease_results,
            "total_found": len(disease_results),
            "tool": "analyze_lab_tests"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "analyze_lab_tests"
        }


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
    Р­РўРђРџ 1: РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РјРµРґРёС†РёРЅСЃРєРѕРіРѕ Р·Р°РїСЂРѕСЃР°
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
    Р­РўРђРџ 2: РџРѕР»СѓС‡РµРЅРёРµ РѕР±Р·РѕСЂРЅРѕР№ РёРЅС„РѕСЂРјР°С†РёРё Рѕ Р·Р°Р±РѕР»РµРІР°РЅРёСЏС…
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
    Р­РўРђРџ 3: РџРѕР»СѓС‡РµРЅРёРµ РєРѕРЅРєСЂРµС‚РЅС‹С… СЂР°Р·РґРµР»РѕРІ РјРµРґРёС†РёРЅСЃРєРѕРіРѕ РґРѕРєСѓРјРµРЅС‚Р°
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
# Sync Management Tools
# --------------------

async def t_get_sync_status() -> Dict[str, Any]:
    """
    РџРѕР»СѓС‡РµРЅРёРµ СЃС‚Р°С‚СѓСЃР° СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёРё СЃ MongoDB
    """
    global _sync_manager

    if _sync_manager is None:
        return {
            "enabled": False,
            "message": "Sync manager not initialized"
        }

    try:
        status = _sync_manager.get_status()
        return {
            "enabled": True,
            "status": status
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e)
        }


async def t_force_sync() -> Dict[str, Any]:
    """
    РџСЂРёРЅСѓРґРёС‚РµР»СЊРЅР°СЏ СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёСЏ РґР°РЅРЅС‹С… РёР· MongoDB
    """
    global _sync_manager

    if _sync_manager is None:
        return {
            "success": False,
            "error": "Sync manager not initialized"
        }

    try:
        result = _sync_manager.force_sync()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
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
        Р­РўРђРџ 1: РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РјРµРґРёС†РёРЅСЃРєРѕРіРѕ Р·Р°РїСЂРѕСЃР°

        РџСЂРµРѕР±СЂР°Р·СѓРµС‚ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРёР№ Р·Р°РїСЂРѕСЃ РІ СЃРїРёСЃРѕРє РєРѕРЅРєСЂРµС‚РЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№.
        Р­С‚Рѕ РџР•Р Р’Р«Р™ РёРЅСЃС‚СЂСѓРјРµРЅС‚, РєРѕС‚РѕСЂС‹Р№ РЅСѓР¶РЅРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ РґР»СЏ Р»СЋР±РѕРіРѕ РјРµРґРёС†РёРЅСЃРєРѕРіРѕ РІРѕРїСЂРѕСЃР°.

        РљРћР“Р”Рђ РРЎРџРћР›Р¬Р—РћР’РђРўР¬:
        - РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ СЃРїСЂР°С€РёРІР°РµС‚ Рѕ СЃРёРјРїС‚РѕРјР°С…, Р·Р°Р±РѕР»РµРІР°РЅРёСЏС…, Р»РµС‡РµРЅРёРё
        - РќСѓР¶РЅРѕ РЅР°Р№С‚Рё РєРѕРЅРєСЂРµС‚РЅС‹Рµ Р±РѕР»РµР·РЅРё РїРѕ РѕРїРёСЃР°РЅРёСЋ
        - РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ СѓРїРѕРјСЏРЅСѓР» РєРѕРґ РњРљР‘-10
        - РќР°С‡Р°Р»Рѕ РјРµРґРёС†РёРЅСЃРєРѕРіРѕ РїРѕРёСЃРєР° РЅРѕРІРѕР№ С‚РµРјС‹

        РџРђР РђРњР•РўР Р«:
        - query: РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅС‹Р№ Р·Р°РїСЂРѕСЃ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ, РЅР°РёРјРµРЅРѕРІР°РЅРёРµ Р±РѕР»РµР·РЅРё (РЅР°РїСЂРёРјРµСЂ: "РїРµСЂРёРєР°СЂРґРёС‚С‹", "СЂРµС‚РёРЅРѕР±Р»Р°СЃС‚РѕРјР°")
        - top_k: СЃРєРѕР»СЊРєРѕ Р·Р°Р±РѕР»РµРІР°РЅРёР№ РЅР°Р№С‚Рё (СЂРµРєРѕРјРµРЅРґСѓРµС‚СЃСЏ 3-5)
        - score_threshold: РјРёРЅРёРјР°Р»СЊРЅР°СЏ СЂРµР»РµРІР°РЅС‚РЅРѕСЃС‚СЊ (0.6 РїРѕРґС…РѕРґРёС‚ РґР»СЏ Р±РѕР»СЊС€РёРЅСЃС‚РІР° СЃР»СѓС‡Р°РµРІ)
        - enable_reranking: СѓР»СѓС‡С€Р°РµС‚ РєР°С‡РµСЃС‚РІРѕ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ (РІСЃРµРіРґР° РёСЃРїРѕР»СЊР·СѓР№ True)
        - rerank_top_k: РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ РјРµРЅСЊС€Рµ РёР»Рё СЂР°РІРЅРѕ top_k

        Р’РћР—Р’Р РђР©РђР•Рў:
        - found_diseases: СЃРїРёСЃРѕРє РЅР°Р№РґРµРЅРЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№ СЃ disease_id Рё РЅР°Р·РІР°РЅРёСЏРјРё
        - has_icd_matches: РЅР°Р№РґРµРЅС‹ Р»Рё С‚РѕС‡РЅС‹Рµ СЃРѕРІРїР°РґРµРЅРёСЏ РїРѕ РњРљР‘-10
        - reranking_applied: РїСЂРёРјРµРЅСЏР»СЃСЏ Р»Рё СЂРµСЂР°РЅРєРёРЅРі РґР»СЏ СѓР»СѓС‡С€РµРЅРёСЏ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ

        РЎРћР’Р•Рў: РџРѕСЃР»Рµ РїРѕР»СѓС‡РµРЅРёСЏ СЃРїРёСЃРєР° Р·Р°Р±РѕР»РµРІР°РЅРёР№ РёСЃРїРѕР»СЊР·СѓР№ medical_get_overview РґР»СЏ РїРѕР»СѓС‡РµРЅРёСЏ РґРµС‚Р°Р»СЊРЅРѕР№ РёРЅС„РѕСЂРјР°С†РёРё.
        РџР РђР’РР›Рћ: РќР• РРЎРџРћР›Р¬Р—РЈР™ Р”Р›РЇ РџРћРРЎРљРђ РРќР¤РћР РњРђР¦РР РџРћ РЈР–Р• РР—Р’Р•РЎРўРќРћРњРЈ Р”РћРљРЈРњР•РќРўРЈ (РЅР°РїСЂРёРјРµСЂ СЃРёРјРїС‚РѕРјС‹ РЅРµРєРѕС‚РѕСЂРѕР№ Р±РѕР»РµР·РЅРё, РєРѕС‚РѕСЂСѓСЋ С‚С‹ СѓР¶Рµ РЅР°С€РµР» СЂР°РЅРµРµ). Р”Р»СЏ С‚Р°РєРѕРіРѕ РїРѕРёСЃРєР° РёСЃРїРѕР»СЊР·СѓР№ medical_get_sections
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
        Р­РўРђРџ 2: РџРѕР»СѓС‡РµРЅРёРµ РѕР±Р·РѕСЂРЅРѕР№ РёРЅС„РѕСЂРјР°С†РёРё Рѕ Р·Р°Р±РѕР»РµРІР°РЅРёСЏС…

        Р’РѕР·РІСЂР°С‰Р°РµС‚ РєСЂР°С‚РєРѕРµ РѕРїРёСЃР°РЅРёРµ Р·Р°Р±РѕР»РµРІР°РЅРёР№ Рё СЃРїРёСЃРѕРє РґРѕСЃС‚СѓРїРЅС‹С… СЂР°Р·РґРµР»РѕРІ РґРѕРєСѓРјРµРЅС‚РѕРІ.
        РСЃРїРѕР»СЊР·СѓР№ РџРћРЎР›Р• medical_normalize_query РґР»СЏ РїРѕР»СѓС‡РµРЅРёСЏ РѕРіР»Р°РІР»РµРЅРёСЏ.

        РљРћР“Р”Рђ РРЎРџРћР›Р¬Р—РћР’РђРўР¬:
        - РџРѕР»СѓС‡РёР» disease_ids РѕС‚ medical_normalize_query
        - РќСѓР¶РЅР° РѕР±С‰Р°СЏ РёРЅС„РѕСЂРјР°С†РёСЏ Рѕ Р·Р°Р±РѕР»РµРІР°РЅРёРё
        - РҐРѕС‡РµС€СЊ СѓР·РЅР°С‚СЊ, РєР°РєРёРµ СЂР°Р·РґРµР»С‹ РґРѕСЃС‚СѓРїРЅС‹ РґР»СЏ РёР·СѓС‡РµРЅРёСЏ

        РџРђР РђРњР•РўР Р«:
        - disease_ids: СЃРїРёСЃРѕРє ID Р·Р°Р±РѕР»РµРІР°РЅРёР№ (РёР· СЂРµР·СѓР»СЊС‚Р°С‚Р° medical_normalize_query)
        - query: РґРѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Р№ РїРѕРёСЃРєРѕРІС‹Р№ Р·Р°РїСЂРѕСЃ РґР»СЏ С„РёР»СЊС‚СЂР°С†РёРё (РЅРµРѕР±СЏР·Р°С‚РµР»СЊРЅРѕ)
        - top_k: РјР°РєСЃРёРјР°Р»СЊРЅРѕРµ РєРѕР»РёС‡РµСЃС‚РІРѕ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ

        Р’РћР—Р’Р РђР©РђР•Рў:
        - found_diseases: СЃРїРёСЃРѕРє СЃ РєСЂР°С‚РєРёРјРё РѕРїРёСЃР°РЅРёСЏРјРё Р·Р°Р±РѕР»РµРІР°РЅРёР№
        - available_sections: РґРѕСЃС‚СѓРїРЅС‹Рµ СЂР°Р·РґРµР»С‹ РґР»СЏ РєР°Р¶РґРѕРіРѕ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ
            * id: РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂ СЂР°Р·РґРµР»Р° РґР»СЏ medical_get_sections
            * title: РЅР°Р·РІР°РЅРёРµ СЂР°Р·РґРµР»Р° (РЅР°РїСЂРёРјРµСЂ, "РЎРёРјРїС‚РѕРјС‹", "Р›РµС‡РµРЅРёРµ")
            * has_content: РµСЃС‚СЊ Р»Рё СЃРѕРґРµСЂР¶РёРјРѕРµ РІ СЂР°Р·РґРµР»Рµ

        РЎРћР’Р•Рў: РР·СѓС‡Рё available_sections Рё РёСЃРїРѕР»СЊР·СѓР№ medical_get_sections РґР»СЏ РїРѕР»СѓС‡РµРЅРёСЏ РєРѕРЅРєСЂРµС‚РЅРѕР№ РёРЅС„РѕСЂРјР°С†РёРё РёР· РёРЅС‚РµСЂРµСЃСѓСЋС‰РёС… СЂР°Р·РґРµР»РѕРІ.
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
        Р­РўРђРџ 3: РџРѕР»СѓС‡РµРЅРёРµ РєРѕРЅРєСЂРµС‚РЅС‹С… СЂР°Р·РґРµР»РѕРІ РјРµРґРёС†РёРЅСЃРєРѕРіРѕ РґРѕРєСѓРјРµРЅС‚Р°
        Р’РѕР·РІСЂР°С‰Р°РµС‚ РїРѕРґСЂРѕР±РЅСѓСЋ РёРЅС„РѕСЂРјР°С†РёСЋ РёР· РєРѕРЅРєСЂРµС‚РЅС‹С… СЂР°Р·РґРµР»РѕРІ РґРѕРєСѓРјРµРЅС‚Р° Рѕ Р·Р°Р±РѕР»РµРІР°РЅРёРё.
        Р­С‚Рѕ Р¤РРќРђР›Р¬РќР«Р™ СЌС‚Р°Рї РґР»СЏ РїРѕР»СѓС‡РµРЅРёСЏ РґРµС‚Р°Р»СЊРЅРѕР№ РјРµРґРёС†РёРЅСЃРєРѕР№ РёРЅС„РѕСЂРјР°С†РёРё.

        РљРћР“Р”Рђ РРЎРџРћР›Р¬Р—РћР’РђРўР¬:
        - РџРѕР»СѓС‡РёР» disease_id РѕС‚ medical_get_overview
        - РќСѓР¶РЅР° РґРµС‚Р°Р»СЊРЅР°СЏ РёРЅС„РѕСЂРјР°С†РёСЏ РёР· РєРѕРЅРєСЂРµС‚РЅС‹С… СЂР°Р·РґРµР»РѕРІ
        - РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ СЃРїСЂР°С€РёРІР°РµС‚ Рѕ СЃРёРјРїС‚РѕРјР°С…, Р»РµС‡РµРЅРёРё, РґРёР°РіРЅРѕСЃС‚РёРєРµ РєРѕРЅРєСЂРµС‚РЅРѕР№ Р±РѕР»РµР·РЅРё
        - РҐРѕС‡РµС€СЊ РїРѕР»СѓС‡РёС‚СЊ РїРѕР»РЅС‹Р№ С‚РµРєСЃС‚ РёР· РјРµРґРёС†РёРЅСЃРєРёС… СЂСѓРєРѕРІРѕРґСЃС‚РІ

        РџРђР РђРњР•РўР Р«:
        - disease_id: ID Р·Р°Р±РѕР»РµРІР°РЅРёСЏ (РёР· medical_get_overview)
        - section_ids: СЃРїРёСЃРѕРє ID СЂР°Р·РґРµР»РѕРІ (РёР· available_sections РІ overview)
            * РџСЂРёРјРµСЂС‹: ["symptoms", "treatment", "diagnosis", "complications"]
            * Р•СЃР»Рё РЅРµ СѓРєР°Р·Р°С‚СЊ - РїРѕР»СѓС‡РёС€СЊ РІСЃРµ РґРѕСЃС‚СѓРїРЅС‹Рµ СЂР°Р·РґРµР»С‹
        - query: СЃРµРјР°РЅС‚РёС‡РµСЃРєРёР№ РїРѕРёСЃРє РІРЅСѓС‚СЂРё СЂР°Р·РґРµР»РѕРІ (РЅР°РїСЂРёРјРµСЂ: "РїРѕР±РѕС‡РЅС‹Рµ СЌС„С„РµРєС‚С‹")
        - top_k: РјР°РєСЃРёРјР°Р»СЊРЅРѕРµ РєРѕР»РёС‡РµСЃС‚РІРѕ СЂР°Р·РґРµР»РѕРІ РІ РѕС‚РІРµС‚Рµ

        Р’РћР—Р’Р РђР©РђР•Рў:
        - disease_id: ID Р·Р°Р±РѕР»РµРІР°РЅРёСЏ
        - canonical_name: РѕС„РёС†РёР°Р»СЊРЅРѕРµ РЅР°Р·РІР°РЅРёРµ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ
        - sections: СЃРїРёСЃРѕРє СЂР°Р·РґРµР»РѕРІ СЃ СЃРѕРґРµСЂР¶РёРјС‹Рј
            * section_id: ID СЂР°Р·РґРµР»Р°
            * section_title: РЅР°Р·РІР°РЅРёРµ СЂР°Р·РґРµР»Р°
            * content: РїРѕР»РЅС‹Р№ С‚РµРєСЃС‚ СЂР°Р·РґРµР»Р°
            * content_length: РґР»РёРЅР° СЃРѕРґРµСЂР¶РёРјРѕРіРѕ
            * score: СЂРµР»РµРІР°РЅС‚РЅРѕСЃС‚СЊ (РµСЃР»Рё РёСЃРїРѕР»СЊР·РѕРІР°Р»СЃСЏ query)

        РЎРўР РђРўР•Р“РР РРЎРџРћР›Р¬Р—РћР’РђРќРРЇ:
        РџРѕР»СѓС‡РёС‚СЊ РІСЃРµ СЂР°Р·РґРµР»С‹:
        medical_get_sections(disease_id="doc_abc123")
        РџРѕР»СѓС‡РёС‚СЊ РєРѕРЅРєСЂРµС‚РЅС‹Рµ СЂР°Р·РґРµР»С‹:
        medical_get_sections(disease_id="doc_abc123", section_ids=["doc_terms", "doc_crat_info_1_1"])
        РЎРµРјР°РЅС‚РёС‡РµСЃРєРёР№ РїРѕРёСЃРє РІРЅСѓС‚СЂРё Р·Р°Р±РѕР»РµРІР°РЅРёСЏ:
        medical_get_sections(disease_id="doc_abc123", query="Р°РЅР°РјРЅРµР·")
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
        РђРЅР°Р»РёР· Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… С‚РµСЃС‚РѕРІ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ РІРѕР·РјРѕР¶РЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№

        РђРЅР°Р»РёР·РёСЂСѓРµС‚ СЂРµР·СѓР»СЊС‚Р°С‚С‹ Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… Р°РЅР°Р»РёР·РѕРІ Рё РІРѕР·РІСЂР°С‰Р°РµС‚ СЃРїРёСЃРѕРє РІРѕР·РјРѕР¶РЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№
        СЃ РѕС†РµРЅРєРѕР№ РІРµСЂРѕСЏС‚РЅРѕСЃС‚Рё РЅР° РѕСЃРЅРѕРІРµ РїР°С‚С‚РµСЂРЅРѕРІ РѕС‚РєР»РѕРЅРµРЅРёР№.

        РљРћР“Р”Рђ РРЎРџРћР›Р¬Р—РћР’РђРўР¬:
        - РЈ РїР°С†РёРµРЅС‚Р° РµСЃС‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚С‹ Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… Р°РЅР°Р»РёР·РѕРІ
        - РќСѓР¶РЅРѕ РѕРїСЂРµРґРµР»РёС‚СЊ РІРѕР·РјРѕР¶РЅС‹Рµ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ РїРѕ РѕС‚РєР»РѕРЅРµРЅРёСЏРј РІ Р°РЅР°Р»РёР·Р°С…
        - РўСЂРµР±СѓРµС‚СЃСЏ РґРёС„С„РµСЂРµРЅС†РёР°Р»СЊРЅР°СЏ РґРёР°РіРЅРѕСЃС‚РёРєР° РЅР° РѕСЃРЅРѕРІРµ Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… РґР°РЅРЅС‹С…

        РџРђР РђРњР•РўР Р«:
        - tests: СЃРїРёСЃРѕРє Р»Р°Р±РѕСЂР°С‚РѕСЂРЅС‹С… С‚РµСЃС‚РѕРІ, РєР°Р¶РґС‹Р№ СЃРѕРґРµСЂР¶РёС‚:
            * name: РЅР°Р·РІР°РЅРёРµ С‚РµСЃС‚Р° (РЅР°РїСЂРёРјРµСЂ: "Р“РµРјРѕРіР»РѕР±РёРЅ", "Р›РµР№РєРѕС†РёС‚С‹")
            * value: Р·РЅР°С‡РµРЅРёРµ С‚РµСЃС‚Р° (РЅР°РїСЂРёРјРµСЂ: "120", "8.5")
            * units: РµРґРёРЅРёС†С‹ РёР·РјРµСЂРµРЅРёСЏ (РЅР°РїСЂРёРјРµСЂ: "Рі/Р»", "Г—10^9/Р»")
        - gender: РїРѕР» РїР°С†РёРµРЅС‚Р° ("male", "female", "unisex") РґР»СЏ СѓС‡РµС‚Р° РЅРѕСЂРј
        - top_k: РјР°РєСЃРёРјР°Р»СЊРЅРѕРµ РєРѕР»РёС‡РµСЃС‚РІРѕ Р·Р°Р±РѕР»РµРІР°РЅРёР№ РІ СЂРµР·СѓР»СЊС‚Р°С‚Рµ (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 10)
        - categories: С„РёР»СЊС‚СЂ РїРѕ РєР°С‚РµРіРѕСЂРёСЏРј Р·Р°Р±РѕР»РµРІР°РЅРёР№ (РЅРµРѕР±СЏР·Р°С‚РµР»СЊРЅРѕ)

        Р’РћР—Р’Р РђР©РђР•Рў:
        - success: СѓСЃРїРµС€РЅРѕСЃС‚СЊ Р°РЅР°Р»РёР·Р°
        - processing_time_ms: РІСЂРµРјСЏ РѕР±СЂР°Р±РѕС‚РєРё РІ РјРёР»Р»РёСЃРµРєСѓРЅРґР°С…
        - results: СЃРїРёСЃРѕРє РЅР°Р№РґРµРЅРЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№ СЃ РѕС†РµРЅРєР°РјРё:
            * disease_id: РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ
            * canonical_name: РѕС„РёС†РёР°Р»СЊРЅРѕРµ РЅР°Р·РІР°РЅРёРµ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ
            * matched_patterns: РєРѕР»РёС‡РµСЃС‚РІРѕ СЃРѕРІРїР°РІС€РёС… РїР°С‚С‚РµСЂРЅРѕРІ
            * total_patterns: РѕР±С‰РµРµ РєРѕР»РёС‡РµСЃС‚РІРѕ РїР°С‚С‚РµСЂРЅРѕРІ РґР»СЏ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ
            * matched_score: Р±Р°Р»Р» Р·Р° СЃРѕРІРїР°РґРµРЅРёСЏ
            * contradiction_penalty: С€С‚СЂР°С„ Р·Р° РїСЂРѕС‚РёРІРѕСЂРµС‡РёСЏ
            * total_score: РёС‚РѕРіРѕРІС‹Р№ Р±Р°Р»Р»
            * normalized_score: РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅС‹Р№ Р±Р°Р»Р» (0-1)
            * matched_details: РґРµС‚Р°Р»Рё СЃРѕРІРїР°РґРµРЅРёР№
            * contradictions: СЃРїРёСЃРѕРє РїСЂРѕС‚РёРІРѕСЂРµС‡РёР№
            * missing_data: РЅРµРґРѕСЃС‚Р°СЋС‰РёРµ РґР°РЅРЅС‹Рµ
        - total_found: РѕР±С‰РµРµ РєРѕР»РёС‡РµСЃС‚РІРѕ РЅР°Р№РґРµРЅРЅС‹С… Р·Р°Р±РѕР»РµРІР°РЅРёР№

        РџР РРњР•Р  РРЎРџРћР›Р¬Р—РћР’РђРќРРЇ:
        tests = [
            {"name": "Р“РµРјРѕРіР»РѕР±РёРЅ", "value": "85", "units": "Рі/Р»"},
            {"name": "Р›РµР№РєРѕС†РёС‚С‹", "value": "12.5", "units": "Г—10^9/Р»"},
            {"name": "РЎРћР­", "value": "45", "units": "РјРј/С‡"}
        ]
        analyze_lab_tests(tests=tests, gender="female", top_k=5)
        """
        return await t_analyze_lab_tests(
            tests=tests,
            gender=gender,
            top_k=top_k,
            categories=categories
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
    print("Lab Analysis Configuration:")
    print(f"  - MongoDB URI: {MONGODB_URI}")
    print(f"  - Database: {MONGODB_DB}")
    print(f"  - Sync interval: {SYNC_INTERVAL}s")
    print()
    print("Medical workflow:")
    print("  1. medical_normalize_query - find diseases by user query (with reranking)")
    print("  2. medical_get_overview - get disease info + available sections")
    print("  3. medical_get_sections - get specific sections content")
    print("  4. analyze_lab_tests - analyze laboratory test results")
    print()
    print("Sync management:")
    print("  5. get_sync_status - check synchronization status")
    print("  6. force_sync - force immediate data sync from MongoDB")
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
