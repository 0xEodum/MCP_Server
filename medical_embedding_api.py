"""
medical_embedding_api.py — эмбеддинги для медицинской системы

Отличия от обычного embedder:
- Оптимизация для медицинских терминов
- Специализированная предобработка текста
- Кэширование для частых запросов
"""
from __future__ import annotations

import re
import hashlib
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalEmbedder:
    """Embedder оптимизированный для медицинских документов."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        normalize_embeddings: bool = True,
        cache_size: int = 1000
    ):
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings
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
        use_cache: bool = False  # Для батчей кэш обычно не эффективен
    ) -> np.ndarray:
        """Получение эмбеддингов для списка текстов."""

        if not texts:
            return np.array([])

        # Предобработка всех текстов
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Получение эмбеддингов
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        return embeddings.astype(np.float32)

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
        results['registry'] = self.encode_single(registry_text)

        # Эмбеддинг для overview (название + краткое содержание)
        overview_parts = [title]

        # Берем первые 3 непустые секции для краткого описания
        content_sections = [s for s in sections if s.get('body', '').strip()][:3]
        for section in content_sections:
            overview_parts.append(section['body'][:200])  # Первые 200 символов

        overview_text = ' '.join(overview_parts)
        results['overview'] = self.encode_single(overview_text)

        # Эмбеддинги для разделов
        results['sections'] = []
        for section in sections:
            if section.get('body', '').strip():
                section_text = f"{section.get('title', '')} {section['body']}"
                section_embedding = self.encode_single(section_text)
                results['sections'].append({
                    'section_id': section.get('id', ''),
                    'embedding': section_embedding
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