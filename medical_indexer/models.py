"""Dataclass models used across the medical indexing stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiseaseRegistryPayload:
    canonical_name: str
    icd10_codes: List[str] = field(default_factory=list)
    disease_id: str = ""
    canonical_name_lc: str = ""


@dataclass
class SectionInfo:
    code: str
    title: str
    has_content: bool = True


@dataclass
class DiseaseOverviewPayload:
    disease_id: str
    canonical_name: str
    icd10_primary: Optional[str] = None
    summary: str = ""
    available_sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DiseaseSectionPayload:
    disease_id: str
    canonical_name: str
    section_id: str
    section_title: str
    content: str
    content_length: int = 0


@dataclass
class MedicalSearchResult:
    disease_id: str
    canonical_name: str
    icd10_codes: List[str]
    synonyms: List[str]
    score: float
    available_sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MedicalOverviewResponse:
    found_diseases: List[Dict[str, Any]]
    total_found: int
    took_ms: int = 0


@dataclass
class MedicalSectionsResponse:
    disease_id: str
    canonical_name: str
    sections: List[Dict[str, Any]]
    total_sections: int
    took_ms: int = 0


@dataclass
class MedicalDocument:
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
