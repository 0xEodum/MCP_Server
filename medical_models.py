"""
medical_models.py — расширение моделей для медицинской системы

Дополнительные модели для работы с медицинскими документами
по архитектуре из arch.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -----------------------------
# Медицинские модели
# -----------------------------

@dataclass
class DiseaseRegistryPayload:
    canonical_name: str
    icd10_codes: List[str] = field(default_factory=list)
    disease_id: str = ""
    canonical_name_lc: str = ""


@dataclass
class SectionInfo:
    """Информация о разделе документа."""

    code: str
    title: str
    has_content: bool = True


@dataclass
class DiseaseOverviewPayload:
    """Payload для коллекции disease_overview."""

    disease_id: str
    canonical_name: str
    icd10_primary: Optional[str] = None
    summary: str = ""
    available_sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DiseaseSectionPayload:
    """Payload для коллекции disease_sections."""

    disease_id: str
    canonical_name: str
    section_id: str  # id из исходного JSON
    section_title: str
    content: str
    content_length: int = 0


@dataclass
class MedicalSearchResult:
    """Результат поиска заболевания."""

    disease_id: str
    canonical_name: str
    icd10_codes: List[str]
    synonyms: List[str]
    score: float
    available_sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MedicalOverviewResponse:
    """Ответ поиска обзорной информации."""

    found_diseases: List[Dict[str, Any]]
    total_found: int
    took_ms: int = 0


@dataclass
class MedicalSectionsResponse:
    """Ответ получения разделов документа."""

    disease_id: str
    canonical_name: str
    sections: List[Dict[str, Any]]
    total_sections: int
    took_ms: int = 0


@dataclass
class MedicalDocument:
    """Входящий JSON документ."""

    doc_title: str
    mkb: List[str]
    chapters: List[str]
    sections: List[Dict[str, Any]]


__all__ = [
    "DiseaseRegistryPayload",
    "SectionInfo",
    "DiseaseOverviewPayload",
    "DiseaseSectionPayload",
    "MedicalSearchResult",
    "MedicalOverviewResponse",
    "MedicalSectionsResponse",
    "MedicalDocument",
]