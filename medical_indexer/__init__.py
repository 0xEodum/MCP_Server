"""Public API for the medical indexer package."""
from .constants import DISEASE_OVERVIEW, DISEASE_REGISTRY, DISEASE_SECTIONS
from .embeddings import MedicalEmbedder, extract_icd10_codes
from .indexing import (
    index_disease_overview,
    index_disease_registry,
    index_disease_sections,
    index_medical_documents,
)
from .models import (
    DiseaseOverviewPayload,
    DiseaseRegistryPayload,
    DiseaseSectionPayload,
    MedicalDocument,
    MedicalOverviewResponse,
    MedicalSearchResult,
    MedicalSectionsResponse,
    SectionInfo,
)
from .qdrant import MedicalQdrantStore
from .search import (
    get_disease_overview,
    get_disease_sections,
    medical_search_workflow,
    normalize_medical_query,
)
from .utils import parse_section_code, slugify_document_title

__all__ = [
    "DISEASE_OVERVIEW",
    "DISEASE_REGISTRY",
    "DISEASE_SECTIONS",
    "MedicalEmbedder",
    "MedicalQdrantStore",
    "MedicalDocument",
    "MedicalSearchResult",
    "MedicalOverviewResponse",
    "MedicalSectionsResponse",
    "DiseaseRegistryPayload",
    "DiseaseOverviewPayload",
    "DiseaseSectionPayload",
    "SectionInfo",
    "extract_icd10_codes",
    "index_medical_documents",
    "index_disease_registry",
    "index_disease_overview",
    "index_disease_sections",
    "normalize_medical_query",
    "get_disease_overview",
    "get_disease_sections",
    "medical_search_workflow",
    "parse_section_code",
    "slugify_document_title",
]
