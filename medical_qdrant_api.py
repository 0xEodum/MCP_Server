"""
medical_qdrant_api.py — Qdrant API специально для медицинской системы

Отличия от обычного qdrant_api:
- Работа с 3 медицинскими коллекциями
- Медицинские фильтры и запросы
- Поддержка поиска по кодам МКБ
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import uuid

# Константы медицинских коллекций
DISEASE_REGISTRY = "disease_registry"
DISEASE_OVERVIEW = "disease_overview"
DISEASE_SECTIONS = "disease_sections"


class MedicalQdrantStore:
    """Qdrant клиент для медицинской системы."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
    ) -> None:
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        else:
            self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)

    # ------------------------
    # Управление коллекциями
    # ------------------------

    def ensure_medical_collections(
        self,
        vector_size: int,
        *,
        distance: rest.Distance = rest.Distance.COSINE,
        recreate: bool = False
    ) -> Dict[str, bool]:
        """Создание всех 3 медицинских коллекций."""

        results = {}

        for collection_name in [DISEASE_REGISTRY, DISEASE_OVERVIEW, DISEASE_SECTIONS]:
            if recreate:
                try:
                    self.client.delete_collection(collection_name)
                except Exception:
                    pass

            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(
                        size=vector_size,
                        distance=distance,
                        on_disk=True
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(
                        indexing_threshold=10_000
                    ),
                )
                results[collection_name] = True
            else:
                results[collection_name] = False

        return results

    def create_medical_indexes(self) -> None:
        for field in ["canonical_name", "icd10_codes", "disease_id", "canonical_name_lc"]:
            self._create_payload_index(DISEASE_REGISTRY, field)
        for field in ["disease_id", "canonical_name", "icd10_primary"]:
            self._create_payload_index(DISEASE_OVERVIEW, field)
        for field in ["disease_id", "section_id"]:
            self._create_payload_index(DISEASE_SECTIONS, field)

    def _create_payload_index(self, collection: str, field: str) -> None:
        """Создание индекса по полю payload."""
        try:
            self.client.create_payload_index(collection, field_name=field)
        except Exception:
            # Индекс уже существует или ошибка создания
            pass

    # ------------------------
    # Upsert операции
    # ------------------------

    @staticmethod
    def _to_float32(vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Приведение к float32."""
        if isinstance(vectors, np.ndarray):
            arr = vectors
        else:
            arr = np.asarray(vectors, dtype=np.float32)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr

    def upsert_to_collection(
        self,
        collection: str,
        *,
        vectors: Sequence[Sequence[float]] | np.ndarray,
        payloads: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[str]] = None,
        wait: bool = True,
    ) -> None:
        """Upsert точек в указанную коллекцию."""

        arr = self._to_float32(vectors)
        if len(arr) != len(payloads):
            raise ValueError("Vectors and payloads length mismatch")
        if ids is not None and len(ids) != len(payloads):
            raise ValueError("IDs length mismatch")

        points: List[rest.PointStruct] = []
        for i in range(len(payloads)):
            # Генерируем UUID для Qdrant - это всегда работает
            if ids is not None:
                # Создаем стабильный UUID5 из переданного ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i]))
            else:
                # Случайный UUID4
                point_id = str(uuid.uuid4())

            vec = arr[i].tolist()
            points.append(rest.PointStruct(
                id=point_id,
                vector=vec,
                payload=payloads[i]
            ))

        self.client.upsert(
            collection_name=collection,
            points=points,
            wait=wait
        )

    # ------------------------
    # Поиск в медицинских коллекциях
    # ------------------------

    def search_diseases_by_vector(
        self,
        query_vector: Sequence[float] | np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.6
    ):
        """Поиск заболеваний в registry по вектору."""
        qv = np.asarray(query_vector, dtype=np.float32).tolist()

        return self.client.search(
            collection_name=DISEASE_REGISTRY,
            query_vector=qv,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )

    def search_diseases_by_icd(self, icd_codes: List[str]):
        """Точный поиск заболеваний по кодам МКБ."""
        icd_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="icd10_codes",
                    match=rest.MatchAny(any=icd_codes)
                )
            ]
        )

        results, _ = self.client.scroll(
            collection_name=DISEASE_REGISTRY,
            scroll_filter=icd_filter,
            limit=20,
            with_payload=True
        )

        return results

    def get_disease_overview(
        self,
        disease_ids: List[str],
        query_vector: Optional[Sequence[float]] = None,
        top_k: int = 5
    ):
        """Получение обзорной информации по disease_ids."""

        disease_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="disease_id",
                    match=rest.MatchAny(any=disease_ids)
                )
            ]
        )

        if query_vector is not None:
            # Семантический поиск с фильтром
            qv = np.asarray(query_vector, dtype=np.float32).tolist()
            return self.client.search(
                collection_name=DISEASE_OVERVIEW,
                query_vector=qv,
                limit=top_k,
                query_filter=disease_filter,
                with_payload=True
            )
        else:
            # Просто получение по фильтру
            results, _ = self.client.scroll(
                collection_name=DISEASE_OVERVIEW,
                scroll_filter=disease_filter,
                limit=top_k,
                with_payload=True
            )
            return results

    def get_disease_sections(
        self,
        disease_id: str,
        section_ids: Optional[List[str]] = None,
        query_vector: Optional[Sequence[float]] = None,
        top_k: int = 10
    ):
        """Получение разделов документа."""

        # Базовый фильтр по disease_id
        filter_conditions = [
            rest.FieldCondition(
                key="disease_id",
                match=rest.MatchValue(value=disease_id)
            )
        ]

        # Дополнительный фильтр по section_ids
        if section_ids:
            filter_conditions.append(
                rest.FieldCondition(
                    key="section_id",
                    match=rest.MatchAny(any=section_ids)
                )
            )

        sections_filter = rest.Filter(must=filter_conditions)

        if query_vector is not None:
            # Семантический поиск по содержимому
            qv = np.asarray(query_vector, dtype=np.float32).tolist()
            return self.client.search(
                collection_name=DISEASE_SECTIONS,
                query_vector=qv,
                limit=top_k,
                query_filter=sections_filter,
                with_payload=True
            )
        else:
            # Получение по фильтрам
            results, _ = self.client.scroll(
                collection_name=DISEASE_SECTIONS,
                scroll_filter=sections_filter,
                limit=top_k,
                with_payload=True
            )
            return results

    def search_diseases_by_name_exact(self, name_lc: str):
        f = rest.Filter(must=[rest.FieldCondition(
            key="canonical_name_lc", match=rest.MatchValue(value=name_lc)
        )])
        results, _ = self.client.scroll(
            collection_name=DISEASE_REGISTRY,
            scroll_filter=f, limit=20, with_payload=True
        )
        return results
    # ------------------------
    # Утилиты
    # ------------------------

    def delete_disease_data(self, disease_id: str) -> Dict[str, bool]:
        """Удаление всех данных заболевания из всех коллекций."""

        results = {}

        # Удаление из registry
        try:
            self.client.delete(
                DISEASE_REGISTRY,
                points_selector=rest.PointIdsList(
                    points=[disease_id]
                )
            )
            results[DISEASE_REGISTRY] = True
        except Exception:
            results[DISEASE_REGISTRY] = False

        # Удаление из overview
        try:
            overview_filter = rest.Filter(
                must=[rest.FieldCondition(
                    key="disease_id",
                    match=rest.MatchValue(value=disease_id)
                )]
            )
            self.client.delete(
                DISEASE_OVERVIEW,
                points_selector=rest.FilterSelector(filter=overview_filter)
            )
            results[DISEASE_OVERVIEW] = True
        except Exception:
            results[DISEASE_OVERVIEW] = False

        # Удаление из sections
        try:
            sections_filter = rest.Filter(
                must=[rest.FieldCondition(
                    key="disease_id",
                    match=rest.MatchValue(value=disease_id)
                )]
            )
            self.client.delete(
                DISEASE_SECTIONS,
                points_selector=rest.FilterSelector(filter=sections_filter)
            )
            results[DISEASE_SECTIONS] = True
        except Exception:
            results[DISEASE_SECTIONS] = False

        return results

    def count_collection(self, collection: str) -> int:
        """Подсчет точек в коллекции."""
        res = self.client.count(collection, exact=False)
        return int(res.count)

    def get_collections_info(self) -> Dict[str, int]:
        """Информация о всех медицинских коллекциях."""
        info = {}
        for collection in [DISEASE_REGISTRY, DISEASE_OVERVIEW, DISEASE_SECTIONS]:
            try:
                info[collection] = self.count_collection(collection)
            except Exception:
                info[collection] = -1
        return info

    def ping(self) -> bool:
        """Проверка подключения."""
        try:
            _ = self.client.get_collections()
            return True
        except Exception:
            return False


__all__ = [
    "MedicalQdrantStore",
    "DISEASE_REGISTRY",
    "DISEASE_OVERVIEW",
    "DISEASE_SECTIONS",
]