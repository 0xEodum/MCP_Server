"""
medical_indexer.py — индексация медицинских документов в Qdrant

Реализует загрузку JSON документов в 3 коллекции согласно arch.md:
- disease_registry (реестр заболеваний с синонимами)
- disease_overview (обзорная информация)
- disease_sections (детальный контент по разделам)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from medical_qdrant_api import MedicalQdrantStore, DISEASE_REGISTRY, DISEASE_OVERVIEW, DISEASE_SECTIONS
from medical_embedding_api import MedicalEmbedder
from medical_models import (
    DiseaseRegistryPayload,
    DiseaseOverviewPayload,
    DiseaseSectionPayload,
    MedicalDocument,
    SectionInfo,
)


# -----------------------------
# Константы коллекций
# -----------------------------

DISEASE_REGISTRY = "disease_registry"
DISEASE_OVERVIEW = "disease_overview"
DISEASE_SECTIONS = "disease_sections"


# -----------------------------
# Утилиты
# -----------------------------

def _slugify(name: str) -> str:
    """Создание безопасного ASCII ID из названия."""
    import hashlib

    # Создаем hash от оригинального названия
    hash_obj = hashlib.md5(name.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]  # Берем первые 12 символов

    # Дополнительно создаем читаемую часть (только ASCII)
    ascii_part = re.sub(r'[^\w]', '_', name.lower(), flags=re.ASCII)
    ascii_part = re.sub(r'_+', '_', ascii_part).strip('_')

    # Если ASCII часть получилась пустой или слишком длинной, используем только hash
    if not ascii_part or len(ascii_part) > 20:
        return f"doc_{hash_hex}"

    return f"{ascii_part}_{hash_hex}"


def _parse_section_code(section_title: str) -> str:
    """Извлечение кода секции из заголовка (например '1.1' из '1.1 Определение')."""
    match = re.match(r'^(\d+(?:\.\d+)*)', section_title.strip())
    return match.group(1) if match else ""


# -----------------------------
# Индексация по коллекциям
# -----------------------------

def index_disease_registry(
    store: MedicalQdrantStore,
    embedder: MedicalEmbedder,
    documents: List[MedicalDocument],
    recreate: bool = False
) -> Dict[str, Any]:
    """Индексация реестра заболеваний."""

    if recreate:
        store.ensure_medical_collections(embedder.get_vector_size(), recreate=True)
    else:
        store.ensure_medical_collections(embedder.get_vector_size(), recreate=False)

    store.create_medical_indexes()

    vectors = []
    payloads = []
    disease_ids = []

    for doc in documents:
        disease_id = _slugify(doc.doc_title)
        embed_text = doc.doc_title
        vector = embedder.encode_single(embed_text)

        payload = DiseaseRegistryPayload(
            canonical_name=doc.doc_title,
            icd10_codes=doc.mkb,
            disease_id=disease_id,  # NEW
            canonical_name_lc=doc.doc_title.lower()  # NEW
        )

        vectors.append(vector)
        payloads.append(asdict(payload))
        disease_ids.append(disease_id)

    if vectors:
        store.upsert_to_collection(
            DISEASE_REGISTRY,
            vectors=vectors,
            payloads=payloads,
            ids=disease_ids
        )

    return {
        "collection": DISEASE_REGISTRY,
        "indexed": len(vectors),
        "disease_ids": disease_ids
    }


def index_disease_overview(
        store: MedicalQdrantStore,
        embedder: MedicalEmbedder,
        documents: List[MedicalDocument],
        recreate: bool = False
) -> Dict[str, Any]:
    """Индексация обзорной информации."""

    # Коллекции уже созданы в index_disease_registry
    vectors = []
    payloads = []
    ids = []

    for doc in documents:
        disease_id = _slugify(doc.doc_title)

        # Создание краткого описания из первых разделов
        summary_parts = []
        for section in doc.sections[:3]:  # первые 3 секции для summary
            if section.get('body') and len(section['body'].strip()) > 10:
                summary_parts.append(section['body'][:200])

        summary = " ".join(summary_parts)[:500] + "..." if summary_parts else ""

        # Доступные разделы
        available_sections = []
        for section in doc.sections:
            if section.get('body') and section.get('title'):
                available_sections.append({
                    "id": section['id'],
                    "title": section['title'],
                    "has_content": len(section['body'].strip()) > 10
                })

        # Эмбеддинг из названия + краткого описания
        embed_text = f"{doc.doc_title} {summary}"
        vector = embedder.encode_single(embed_text)

        payload = DiseaseOverviewPayload(
            disease_id=disease_id,
            canonical_name=doc.doc_title,
            icd10_primary=doc.mkb[0] if doc.mkb else None,
            summary=summary,
            available_sections=available_sections
        )

        vectors.append(vector)
        payloads.append(asdict(payload))
        ids.append(f"{disease_id}_overview")

    if vectors:
        print(f"Индексируем {len(vectors)} записей в {DISEASE_OVERVIEW}...")
        store.upsert_to_collection(
            DISEASE_OVERVIEW,
            vectors=vectors,
            payloads=payloads,
            ids=ids
        )

    return {
        "collection": DISEASE_OVERVIEW,
        "indexed": len(vectors)
    }


def index_disease_sections(
        store: MedicalQdrantStore,
        embedder: MedicalEmbedder,
        documents: List[MedicalDocument],
        recreate: bool = False
) -> Dict[str, Any]:
    """Индексация детальных разделов."""

    # Коллекции уже созданы в index_disease_registry
    vectors = []
    payloads = []
    ids = []
    total_sections = 0

    for doc in documents:
        disease_id = _slugify(doc.doc_title)

        for section in doc.sections:
            body = section.get('body', '').strip()
            title = section.get('title', '').strip()
            section_id = section.get('id', '')

            # Пропускаем пустые секции
            if not body or len(body) < 10:
                continue

            # Эмбеддинг из заголовка + содержимого
            embed_text = f"{title} {body}"
            vector = embedder.encode_single(embed_text)

            payload = DiseaseSectionPayload(
                disease_id=disease_id,
                canonical_name=doc.doc_title,
                section_id=section_id,
                section_title=title,
                content=body,
                content_length=len(body)
            )

            vectors.append(vector)
            payloads.append(asdict(payload))
            ids.append(f"{disease_id}#{section_id}")
            total_sections += 1

    if vectors:
        print(f"Индексируем {len(vectors)} записей в {DISEASE_SECTIONS}...")
        # Разбиваем на батчи для больших объемов
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            batch_num = i // batch_size + 1
            total_batches = (len(vectors) + batch_size - 1) // batch_size

            if total_batches > 1:
                print(f"  Батч {batch_num}/{total_batches}: {len(batch_vectors)} записей")

            store.upsert_to_collection(
                DISEASE_SECTIONS,
                vectors=batch_vectors,
                payloads=batch_payloads,
                ids=batch_ids
            )

    return {
        "collection": DISEASE_SECTIONS,
        "indexed": total_sections
    }


# -----------------------------
# Главная функция индексации
# -----------------------------

def index_medical_documents(
    store: MedicalQdrantStore,
    embedder: MedicalEmbedder,
    json_files: List[str | Path],
    recreate_collections: bool = False
) -> Dict[str, Any]:
    """Индексация медицинских документов из JSON файлов."""

    documents = []

    # Загрузка всех JSON файлов
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                doc = MedicalDocument(
                    doc_title=data['doc_title'],
                    mkb=data.get('mkb', []),
                    chapters=data.get('chapters', []),
                    sections=data.get('sections', [])
                )
                documents.append(doc)
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            continue

    if not documents:
        return {"error": "Не загружено ни одного документа"}

    results = {}

    # Индексация в 3 коллекции
    print(f"Индексация {len(documents)} документов...")

    results['registry'] = index_disease_registry(store, embedder, documents, recreate_collections)
    print(f"✓ Registry: {results['registry']['indexed']} записей")

    results['overview'] = index_disease_overview(store, embedder, documents, recreate_collections)
    print(f"✓ Overview: {results['overview']['indexed']} записей")

    results['sections'] = index_disease_sections(store, embedder, documents, recreate_collections)
    print(f"✓ Sections: {results['sections']['indexed']} записей")

    results['summary'] = {
        "total_documents": len(documents),
        "collections_created": 3,
        "total_vectors": (
            results['registry']['indexed'] +
            results['overview']['indexed'] +
            results['sections']['indexed']
        )
    }

    return results


__all__ = [
    "index_medical_documents",
    "index_disease_registry",
    "index_disease_overview",
    "index_disease_sections",
    "DISEASE_REGISTRY",
    "DISEASE_OVERVIEW",
    "DISEASE_SECTIONS",
]