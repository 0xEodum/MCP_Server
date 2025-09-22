"""
medical_embedding_api.py — эмбеддинги для медицинской системы

Отличия от обычного embedder:
- Оптимизация для медицинских терминов
- Специализированная предобработка текста
- Кэширование для частых запросов
- Реранкинг с помощью cross-encoder для повышения качества
"""
from __future__ import annotations

import re
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class MedicalEmbedder:
    """Embedder оптимизированный для медицинских документов."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        normalize_embeddings: bool = True,
        cache_size: int = 1000
    ):
        # Основная модель для эмбеддингов
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

        # Модель для реранкинга
        self.reranker = CrossEncoder(reranker_model)

        # Кэш
        self.cache: Dict[str, List[float]] = {}
        self.cache_size = cache_size

    def get_vector_size(self) -> int:
        """Размерность векторов модели."""
        return self.model.get_sentence_embedding_dimension()

    def preprocess_text(self, text: str) -> str:
        """Базовая предобработка текста."""
        # Нормализация пробелов и переносов
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def _get_cache_key(self, text: str) -> str:
        """Создание ключа кэша для текста."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def encode_single(self, text: str, use_cache: bool = True) -> List[float]:
        """Получение эмбеддинга для одного текста."""

        # Базовая предобработка
        processed_text = self.preprocess_text(text)

        # Для E5 моделей добавляем префикс для query
        if "e5" in self.model._modules['0'].auto_model.name_or_path.lower():
            if not processed_text.startswith("query:"):
                processed_text = f"query: {processed_text}"

        # Проверка кэша
        if use_cache:
            cache_key = self._get_cache_key(processed_text)
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Получение эмбеддинга
        embedding = self.model.encode(
            processed_text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        vector = embedding.astype(np.float32).tolist()

        # Сохранение в кэш
        if use_cache and len(self.cache) < self.cache_size:
            self.cache[cache_key] = vector

        return vector

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
        use_cache: bool = False,  # Для батчей кэш обычно не эффективен
        add_prefix: bool = True
    ) -> np.ndarray:
        """Получение эмбеддингов для списка текстов."""

        if not texts:
            return np.array([])

        # Предобработка всех текстов
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Для E5 моделей добавляем префиксы
        if add_prefix and "e5" in self.model._modules['0'].auto_model.name_or_path.lower():
            processed_texts = [f"passage: {text}" if not text.startswith(("query:", "passage:"))
                             else text for text in processed_texts]

        # Получение эмбеддингов
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        return embeddings.astype(np.float32)

    def rerank_results(
        self,
        query: str,
        candidates: List[Dict],
        text_field: str = "canonical_name",
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Реранкинг результатов с помощью cross-encoder.

        Args:
            query: Исходный запрос
            candidates: Список кандидатов с полем text_field
            text_field: Поле для сравнения с запросом
            top_k: Количество результатов после реранкинга

        Returns:
            Отсортированный список кандидатов с обновленными scores
        """
        if not candidates:
            return candidates

        # Подготовка пар (query, candidate) для реранкера
        pairs = []
        for candidate in candidates:
            candidate_text = candidate.get(text_field, "")
            if candidate_text:
                pairs.append([query, candidate_text])
            else:
                pairs.append([query, ""])

        # Получение скоров от реранкера
        if pairs:
            rerank_scores = self.reranker.predict(pairs)

            # Обновление скоров в результатах
            reranked_candidates = []
            for i, candidate in enumerate(candidates):
                updated_candidate = candidate.copy()
                updated_candidate['rerank_score'] = float(rerank_scores[i])
                updated_candidate['original_score'] = candidate.get('score', 0.0)
                reranked_candidates.append(updated_candidate)

            # Сортировка по новому скору
            reranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

            # Обновление основного скора на реранк скор
            for candidate in reranked_candidates:
                candidate['score'] = candidate['rerank_score']

            # Ограничение количества результатов
            if top_k:
                reranked_candidates = reranked_candidates[:top_k]

            return reranked_candidates

        return candidates

    def encode_medical_document(
        self,
        title: str,
        sections: List[Dict[str, str]],
        include_icd_codes: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Специализированное кодирование медицинского документа."""

        results = {}

        # Эмбеддинг для registry (название + коды МКБ)
        registry_text = title
        if include_icd_codes:
            registry_text += f" {' '.join(include_icd_codes)}"

        # Добавляем префикс для E5 модели
        if "e5" in self.model._modules['0'].auto_model.name_or_path.lower():
            registry_text = f"passage: {registry_text}"

        results['registry'] = self.encode_single(registry_text, use_cache=False)

        # Эмбеддинг для overview (название + краткое содержание)
        overview_parts = [title]

        # Берем первые 3 непустые секции для краткого описания
        content_sections = [s for s in sections if s.get('body', '').strip()][:3]
        for section in content_sections:
            overview_parts.append(section['body'][:200])  # Первые 200 символов

        overview_text = ' '.join(overview_parts)
        if "e5" in self.model._modules['0'].auto_model.name_or_path.lower():
            overview_text = f"passage: {overview_text}"

        results['overview'] = self.encode_single(overview_text, use_cache=False)

        # Эмбеддинги для разделов
        results['sections'] = []
        section_texts = []
        section_metas = []

        for section in sections:
            if section.get('body', '').strip():
                section_text = f"{section.get('title', '')} {section['body']}"
                section_texts.append(section_text)
                section_metas.append({
                    'section_id': section.get('id', ''),
                    'title': section.get('title', '')
                })

        # Батчевое кодирование секций
        if section_texts:
            section_embeddings = self.encode_batch(section_texts, add_prefix=True)

            for i, embedding in enumerate(section_embeddings):
                results['sections'].append({
                    'section_id': section_metas[i]['section_id'],
                    'embedding': embedding.tolist()
                })

        return results

    def clear_cache(self) -> None:
        """Очистка кэша эмбеддингов."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Статистика кэша."""
        return {
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size
        }


# Утилитарные функции
def extract_icd10_codes(text: str) -> List[str]:
    """Извлечение кодов МКБ-10 из текста."""
    # Паттерн для кодов МКБ-10: буква + 2 цифры + точка + 1-2 цифры (опционально)
    pattern = r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'
    return list(set(re.findall(pattern, text.upper())))


__all__ = [
    "MedicalEmbedder",
    "extract_icd10_codes",
]