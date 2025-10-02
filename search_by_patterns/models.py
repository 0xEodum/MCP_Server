"""
Data models for Medical Lab Disease Search Engine
Модели данных для поискового движка заболеваний
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestResult:
    """Результат лабораторного теста"""
    name: str
    value: float
    units: str
    status: Optional[str] = None
    category: Optional[str] = None


@dataclass
class Pattern:
    """Паттерн отклонения для заболевания"""
    test_name: str
    expected_status: str
    category: str
    idf_weight: float = 1.0


@dataclass
class Disease:
    """Заболевание с паттернами"""
    disease_id: str
    canonical_name: str
    patterns: List[Pattern] = field(default_factory=list)
    max_idf_score: float = 0.0

    def calculate_max_score(self):
        """Расчёт максимального возможного скора"""
        self.max_idf_score = sum(p.idf_weight for p in self.patterns)


@dataclass
class SearchResult:
    """Результат поиска заболевания"""
    disease_id: str
    canonical_name: str
    matched_patterns: int
    total_patterns: int
    matched_score: float
    contradiction_penalty: float
    total_score: float
    max_possible_score: float
    normalized_score: float
    matched_details: List[Dict]
    contradictions: List[Dict]
    missing_data: List[Dict]