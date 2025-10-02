"""
Medical Lab Disease Search Engine
Поисковый движок для определения заболеваний по лабораторным анализам
"""

import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import re


class UnitConverter:
    """Converts numeric values between compatible laboratory units."""

    _UNIT_ALIASES = {
        '': '',
        '%': 'percent',
        'percent': 'percent',
        'g/l': 'g_per_l',
        'g/dl': 'g_per_dl',
        'mg/l': 'mg_per_l',
        'mg/dl': 'mg_per_dl',
        'mmol/l': 'mmol_per_l',
        'umol/l': 'umol_per_l',
        'µmol/l': 'umol_per_l',
        'mol/l': 'mol_per_l',
        '10^12/l': '1e12_per_l',
        'x10^12/l': '1e12_per_l',
        '10^6/ul': '1e12_per_l',
        'x10^6/ul': '1e12_per_l',
        '10^6/µl': '1e12_per_l',
        'x10^6/µl': '1e12_per_l',
        '10^9/l': '1e9_per_l',
        'x10^9/l': '1e9_per_l',
        '10^3/ul': '1e9_per_l',
        'x10^3/ul': '1e9_per_l',
        '10^3/µl': '1e9_per_l',
        'x10^3/µl': '1e9_per_l',
        'unit': 'unit',
        'units': 'unit',
        'pg': 'pg',
        'fl': 'fl',
        'mm/h': 'mm_per_hr',
        'mm/hr': 'mm_per_hr',
    }

    _CONVERSION_TABLE = {
        ('g_per_l', 'g_per_dl'): 0.1,
        ('g_per_dl', 'g_per_l'): 10.0,
        ('mg_per_l', 'mg_per_dl'): 0.1,
        ('mg_per_dl', 'mg_per_l'): 10.0,
        ('g_per_l', 'mg_per_dl'): 100.0,
        ('mg_per_dl', 'g_per_l'): 0.01,
        ('mmol_per_l', 'umol_per_l'): 1000.0,
        ('umol_per_l', 'mmol_per_l'): 0.001,
        ('1e12_per_l', '1e12_per_l'): 1.0,
        ('1e9_per_l', '1e9_per_l'): 1.0,
        ('percent', 'percent'): 1.0,
        ('unit', 'unit'): 1.0,
        ('pg', 'pg'): 1.0,
        ('fl', 'fl'): 1.0,
        ('mm_per_hr', 'mm_per_hr'): 1.0,
    }

    _REPLACEMENTS = [
        ('×', 'x'),
        ('х', 'x'),
        ('−', '-'),
        ('–', '-'),
        ('µl', 'ul'),
        ('μl', 'ul'),
        ('µ', 'u'),
        ('μ', 'u'),
        ('мкмоль', 'umol'),
        ('ммоль', 'mmol'),
        ('моль', 'mol'),
        ('мкл', 'ul'),
        ('мл', 'ml'),
        ('мк', 'u'),
        ('мм', 'mm'),
        ('мг', 'mg'),
        ('г', 'g'),
        ('дл', 'dl'),
        ('л', 'l'),
        ('ед.', 'unit'),
        ('ед', 'unit'),
        ('ч', 'h'),
        ('пг', 'pg'),
        ('фл', 'fl'),
        (',', '.'),
    ]

    def normalize_unit(self, unit: Optional[str]) -> str:
        if not unit:
            return ''

        normalized = unit.strip().lower()
        if not normalized:
            return ''

        for src, dst in self._REPLACEMENTS:
            normalized = normalized.replace(src, dst)

        normalized = normalized.replace(' ', '')

        return self._UNIT_ALIASES.get(normalized, normalized)

    def convert(self, value: float, from_unit: Optional[str], to_unit: Optional[str]) -> float:
        from_key = self.normalize_unit(from_unit)
        to_key = self.normalize_unit(to_unit)

        if not from_key or not to_key or from_key == to_key:
            return value

        factor = self._CONVERSION_TABLE.get((from_key, to_key))
        if factor is not None:
            return value * factor

        inverse = self._CONVERSION_TABLE.get((to_key, from_key))
        if inverse is not None and inverse != 0:
            return value / inverse

        return value

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


class ReferenceRangeManager:
    """Управление референсными диапазонами"""
    
    def __init__(self):
        self.references: Dict[str, Dict] = {}
        # Индекс: normalized_name -> (category, original_name)
        self.name_index: Dict[str, Tuple[str, str]] = {}
        self.unit_converter = UnitConverter()
    
    def load_from_json(self, json_path: str):
        """Загрузка референсов из JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ref_ranges = data.get('reference_ranges', {})
        
        for category, tests in ref_ranges.items():
            if category not in self.references:
                self.references[category] = {}
            
            for test in tests:
                test_name = test['test_name']
                self.references[category][test_name] = test
                
                # Индексируем основное имя
                norm_name = self._normalize(test_name)
                self.name_index[norm_name] = (category, test_name)
                
                # Индексируем альтернативные имена
                for alt_name in test.get('alt_names', []):
                    norm_alt = self._normalize(alt_name)
                    self.name_index[norm_alt] = (category, test_name)
        
        print(f"✓ Loaded {sum(len(tests) for tests in self.references.values())} reference ranges")
        print(f"✓ Built name index with {len(self.name_index)} entries")
    
    def find_test(self, test_name: str) -> Optional[Tuple[str, Dict]]:
        """
        Поиск теста по имени (включая альтернативные названия)
        Возвращает: (category, test_data) или None
        """
        norm_name = self._normalize(test_name)
        
        if norm_name in self.name_index:
            category, original_name = self.name_index[norm_name]
            return category, self.references[category][original_name]
        
        return None
    
    def calculate_status(
        self,
        test_name: str,
        value: float,
        gender: str = 'unisex',
        units: Optional[str] = None
    ) -> str:
        """
        Определение статуса теста
        Возвращает: 'normal', 'below_normal', 'above_normal', 'critically_low', 'critically_high', 'unknown'
        """
        test_info = self.find_test(test_name)
        if not test_info:
            return 'unknown'
        
        _, test_data = test_info
        # Convert patient measurement to reference units when possible
        target_units = test_data.get('units')
        value = self.unit_converter.convert(value, units, target_units)
        
        # ПРИОРИТЕТ 1: Абсолютные диапазоны статусов
        status_ranges = test_data.get('status_ranges', {})
        if status_ranges:
            # Определяем какой диапазон использовать
            if gender in status_ranges:
                ranges = status_ranges[gender]
            elif 'unisex' in status_ranges:
                ranges = status_ranges['unisex']
            else:
                # Берём первый доступный
                ranges = next(iter(status_ranges.values()))
            
            # Проверяем критически низкий
            if 'critically_low' in ranges:
                if 'max' in ranges['critically_low'] and value <= ranges['critically_low']['max']:
                    return 'critically_low'
            
            # Проверяем ниже нормы
            if 'below_normal' in ranges:
                rng = ranges['below_normal']
                if 'min' in rng and 'max' in rng:
                    if rng['min'] <= value < rng['max']:
                        return 'below_normal'
            
            # Проверяем норму
            if 'normal' in ranges:
                rng = ranges['normal']
                if 'min' in rng and 'max' in rng:
                    if rng['min'] <= value <= rng['max']:
                        return 'normal'
            
            # Проверяем выше нормы
            if 'above_normal' in ranges:
                rng = ranges['above_normal']
                if 'min' in rng and 'max' in rng:
                    if rng['min'] < value <= rng['max']:
                        return 'above_normal'
            
            # Проверяем критически высокий
            if 'critically_high' in ranges:
                if 'min' in ranges['critically_high'] and value >= ranges['critically_high']['min']:
                    return 'critically_high'
        
        # ПРИОРИТЕТ 2: Процентные пороги от нормального диапазона
        normal_range = test_data.get('normal_range', {})
        
        # Определяем диапазон по полу
        if isinstance(normal_range, dict):
            if gender in normal_range:
                range_data = normal_range[gender]
            elif 'unisex' in normal_range:
                range_data = normal_range['unisex']
            elif 'min' in normal_range and 'max' in normal_range:
                range_data = normal_range
            else:
                range_data = next(iter(normal_range.values()), {})
        else:
            return 'unknown'
        
        min_val = range_data.get('min')
        max_val = range_data.get('max')
        
        if min_val is None or max_val is None:
            return 'unknown'
        
        # Пороги отклонений
        thresholds = test_data.get('deviation_thresholds', {})
        mild_pct = thresholds.get('mild_deviation_pct', 10)
        significant_pct = thresholds.get('significant_deviation_pct', 30)
        
        # Проверка нормы
        if min_val <= value <= max_val:
            return 'normal'
        
        # Ниже нормы
        if value < min_val:
            deviation_pct = ((min_val - value) / min_val) * 100
            
            if deviation_pct <= mild_pct:
                return 'normal'
            elif deviation_pct <= significant_pct:
                return 'below_normal'
            else:
                return 'critically_low'
        
        # Выше нормы
        if value > max_val:
            deviation_pct = ((value - max_val) / max_val) * 100
            
            if deviation_pct <= mild_pct:
                return 'normal'
            elif deviation_pct <= significant_pct:
                return 'above_normal'
            else:
                return 'critically_high'
        
        return 'unknown'
    
    @staticmethod
    def _normalize(name: str) -> str:
        """Нормализация названия теста"""
        # Убираем лишние пробелы, приводим к нижнему регистру
        normalized = name.lower().strip()
        # Заменяем множественные пробелы на один
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized


class IDFCalculator:
    """Расчёт IDF весов для паттернов"""
    
    @staticmethod
    def calculate_idf_weights(diseases: List[Disease]) -> None:
        """
        Расчёт IDF весов для всех паттернов
        Изменяет объекты Disease in-place
        """
        total_diseases = len(diseases)
        
        if total_diseases == 0:
            return
        
        # Подсчёт document frequency для каждого паттерна
        pattern_df: Dict[str, int] = defaultdict(int)
        
        for disease in diseases:
            # Уникальные паттерны в заболевании
            unique_patterns = set()
            for pattern in disease.patterns:
                pattern_key = IDFCalculator._make_pattern_key(
                    pattern.test_name,
                    pattern.expected_status
                )
                unique_patterns.add(pattern_key)
            
            # Увеличиваем DF для каждого уникального паттерна
            for pattern_key in unique_patterns:
                pattern_df[pattern_key] += 1
        
        # Расчёт IDF для каждого паттерна
        for disease in diseases:
            for pattern in disease.patterns:
                pattern_key = IDFCalculator._make_pattern_key(
                    pattern.test_name,
                    pattern.expected_status
                )
                df = pattern_df[pattern_key]
                
                # IDF = ln((N + 1) / (DF + 1))
                pattern.idf_weight = math.log((total_diseases + 1) / (df + 1))
            
            # Пересчитываем максимальный скор
            disease.calculate_max_score()
        
        print(f"✓ Calculated IDF weights for {len(pattern_df)} unique patterns")
        print(f"  Total diseases: {total_diseases}")
        avg_idf = sum(
            p.idf_weight for d in diseases for p in d.patterns
        ) / sum(len(d.patterns) for d in diseases)
        print(f"  Average IDF weight: {avg_idf:.4f}")
    
    @staticmethod
    def _make_pattern_key(test_name: str, status: str) -> str:
        """Создание ключа паттерна"""
        normalized = ReferenceRangeManager._normalize(test_name)
        return f"{normalized}:{status}"


class DiseaseSearchEngine:
    """
    Поисковый движок для определения заболеваний
    Работает полностью in-memory с инвертированным индексом
    """
    
    def __init__(self, reference_manager: ReferenceRangeManager):
        self.reference_manager = reference_manager
        self.diseases: Dict[str, Disease] = {}
        
        # Инвертированный индекс: pattern_key -> [(disease_id, idf_weight, category)]
        self.pattern_index: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
        
        # Индекс по категориям: category -> [disease_ids]
        self.category_index: Dict[str, set] = defaultdict(set)
    
    def load_diseases_from_json(self, json_path: str):
        """Загрузка заболеваний из JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        diseases_data = data.get('diseases', [])
        diseases_list = []
        
        for disease_data in diseases_data:
            disease = Disease(
                disease_id=disease_data['disease_id'],
                canonical_name=disease_data['canonical_name'],
                patterns=[]
            )
            
            # Загружаем паттерны по категориям
            pattern_groups = disease_data.get('pattern_groups', {})
            for category, patterns in pattern_groups.items():
                for pattern_data in patterns:
                    pattern = Pattern(
                        test_name=pattern_data['test_name'],
                        expected_status=pattern_data['status'],
                        category=category
                    )
                    disease.patterns.append(pattern)
            
            diseases_list.append(disease)
        
        # Расчёт IDF весов
        IDFCalculator.calculate_idf_weights(diseases_list)
        
        # Сохраняем в словарь и строим индексы
        for disease in diseases_list:
            self.diseases[disease.disease_id] = disease
            self._index_disease(disease)
        
        print(f"✓ Loaded {len(self.diseases)} diseases into search engine")
    
    def _index_disease(self, disease: Disease):
        """Построение индексов для заболевания"""
        for pattern in disease.patterns:
            # Находим каноническое название теста через reference manager
            test_info = self.reference_manager.find_test(pattern.test_name)
            if test_info:
                _, test_data = test_info
                canonical_name = test_data['test_name']
            else:
                canonical_name = pattern.test_name
            
            pattern_key = self._make_pattern_key(
                canonical_name,
                pattern.expected_status
            )
            
            self.pattern_index[pattern_key].append((
                disease.disease_id,
                pattern.idf_weight,
                pattern.category
            ))
            
            self.category_index[pattern.category].add(disease.disease_id)
    
    def search(
        self,
        patient_tests: List[TestResult],
        top_k: int = 10,
        gender: str = 'unisex',
        categories: Optional[List[str]] = None,
        min_matched_patterns: int = 1,
        apply_contradiction_penalty: bool = True
    ) -> List[SearchResult]:
        """
        Поиск заболеваний по результатам анализов пациента
        
        Args:
            patient_tests: Список результатов анализов
            top_k: Количество результатов
            gender: Пол пациента ('male', 'female', 'unisex')
            categories: Фильтр по категориям (None = все категории)
            min_matched_patterns: Минимум совпадений для включения в результаты
            apply_contradiction_penalty: Применять ли штраф за противоречия
        
        Returns:
            Список результатов, отсортированный по релевантности
        """
        # 1. Определяем статусы для всех тестов пациента
        patient_patterns: Dict[str, Tuple[str, str, str]] = {}  # normalized_name -> (status, category, canonical_name)
        
        for test in patient_tests:
            status = self.reference_manager.calculate_status(
                test.name,
                test.value,
                gender,
                test.units
            )
            
            if status == 'unknown':
                continue
            
            # Находим категорию теста и каноническое название
            test_info = self.reference_manager.find_test(test.name)
            if test_info:
                category, test_data = test_info
                canonical_name = test_data['test_name']
            else:
                category = 'unknown'
                canonical_name = test.name
            
            normalized_name = self._normalize(canonical_name)
            patient_patterns[normalized_name] = (status, category, canonical_name)
            test.status = status
            test.category = category
        
        if not patient_patterns:
            return []
        
        # 2. Собираем релевантные заболевания через инвертированный индекс
        disease_scores: Dict[str, Dict] = defaultdict(lambda: {
            'matched_score': 0.0,
            'contradiction_penalty': 0.0,
            'matched_patterns': [],
            'contradictions': [],
            'missing_data': []
        })
        
        # Проходим по паттернам пациента
        for test_name, (status, category, canonical_name) in patient_patterns.items():
            # Фильтр по категориям
            if categories and category not in categories:
                continue
            
            pattern_key = self._make_pattern_key(canonical_name, status)
            
            # Находим все заболевания с этим паттерном (O(1) lookup!)
            if pattern_key in self.pattern_index:
                for disease_id, idf_weight, pattern_category in self.pattern_index[pattern_key]:
                    # Фильтр по категориям заболевания
                    if categories and pattern_category not in categories:
                        continue
                    
                    # Это совпадение!
                    disease_scores[disease_id]['matched_score'] += idf_weight
                    disease_scores[disease_id]['matched_patterns'].append({
                        'test_name': test_name,
                        'status': status,
                        'idf_weight': idf_weight,
                        'category': pattern_category
                    })
        
        # 3. Проверяем противоречия (только для найденных заболеваний)
        for disease_id in disease_scores.keys():
            disease = self.diseases[disease_id]
            
            for pattern in disease.patterns:
                # Находим каноническое название
                test_info = self.reference_manager.find_test(pattern.test_name)
                if test_info:
                    _, test_data = test_info
                    canonical_name = test_data['test_name']
                else:
                    canonical_name = pattern.test_name
                
                normalized_test = self._normalize(canonical_name)
                
                # Есть ли этот тест у пациента?
                if normalized_test in patient_patterns:
                    patient_status, _, _ = patient_patterns[normalized_test]
                    expected_status = pattern.expected_status
                    
                    # Противоречие?
                    if patient_status != expected_status:
                        if apply_contradiction_penalty:
                            disease_scores[disease_id]['contradiction_penalty'] += pattern.idf_weight
                        
                        disease_scores[disease_id]['contradictions'].append({
                            'test_name': pattern.test_name,
                            'expected': expected_status,
                            'actual': patient_status,
                            'penalty': pattern.idf_weight,
                            'category': pattern.category
                        })
                else:
                    # Отсутствующие данные
                    disease_scores[disease_id]['missing_data'].append({
                        'test_name': pattern.test_name,
                        'expected_status': pattern.expected_status,
                        'reason': 'not_in_panel',
                        'idf_weight': pattern.idf_weight,
                        'category': pattern.category
                    })
        
        # 4. Финальный скоринг и формирование результатов
        results = []
        
        for disease_id, scores in disease_scores.items():
            if len(scores['matched_patterns']) < min_matched_patterns:
                continue
            
            disease = self.diseases[disease_id]
            
            matched_score = scores['matched_score']
            contradiction_penalty = scores['contradiction_penalty']
            total_score = matched_score - contradiction_penalty
            max_score = disease.max_idf_score
            
            result = SearchResult(
                disease_id=disease_id,
                canonical_name=disease.canonical_name,
                matched_patterns=len(scores['matched_patterns']),
                total_patterns=len(disease.patterns),
                matched_score=matched_score,
                contradiction_penalty=-contradiction_penalty,
                total_score=total_score,
                max_possible_score=max_score,
                normalized_score=total_score / max_score if max_score > 0 else 0.0,
                matched_details=scores['matched_patterns'],
                contradictions=scores['contradictions'],
                missing_data=scores['missing_data']
            )
            
            results.append(result)
        
        # 5. Сортировка по релевантности
        results.sort(
            key=lambda x: (x.total_score, x.normalized_score, x.matched_patterns),
            reverse=True
        )
        
        return results[:top_k]
    
    def _make_pattern_key(self, test_name: str, status: str) -> str:
        """Создание ключа паттерна"""
        normalized = self._normalize(test_name)
        return f"{normalized}:{status}"
    
    @staticmethod
    def _normalize(name: str) -> str:
        """Нормализация названия теста"""
        return ReferenceRangeManager._normalize(name)


class MedicalLabAnalyzer:
    """
    Главный класс для анализа лабораторных тестов
    Объединяет все компоненты системы
    Поддерживает загрузку как из JSON файлов, так и из MongoDB
    """
    
    def __init__(self, mongodb_client=None):
        """
        Args:
            mongodb_client: Опциональный MongoClient для работы с MongoDB
        """
        self.reference_manager = ReferenceRangeManager()
        self.search_engine = None
        self.mongodb_client = mongodb_client
        self.mongodb_db = None
        
        if mongodb_client:
            self.mongodb_db = mongodb_client.medical_lab
    
    def load_references(self, references_path: str):
        """Загрузка референсных диапазонов"""
        print("=" * 60)
        print("Loading reference ranges...")
        self.reference_manager.load_from_json(references_path)
        print("=" * 60)
    
    def load_diseases(self, diseases_path: str):
        """Загрузка базы заболеваний"""
        print("=" * 60)
        print("Loading disease database...")
        self.search_engine = DiseaseSearchEngine(self.reference_manager)
        self.search_engine.load_diseases_from_json(diseases_path)
        print("=" * 60)
    
    def analyze_patient(
        self,
        tests: List[Dict],
        gender: str = 'unisex',
        top_k: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Анализ результатов пациента
        
        Args:
            tests: Список тестов вида [{"name": "...", "value": "...", "units": "..."}]
            gender: Пол пациента
            top_k: Количество результатов
            categories: Фильтр по категориям анализов
        """
        if not self.search_engine:
            raise ValueError("Disease database not loaded. Call load_diseases() first.")
        
        # Преобразуем входные данные в TestResult
        test_results = []
        for test in tests:
            try:
                value = float(test['value'])
                test_result = TestResult(
                    name=test['name'],
                    value=value,
                    units=test.get('units', '')
                )
                test_results.append(test_result)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid test: {test}. Error: {e}")
                continue
        
        # Поиск заболеваний
        results = self.search_engine.search(
            patient_tests=test_results,
            top_k=top_k,
            gender=gender,
            categories=categories
        )
        
        return results
    
    def print_results(self, results: List[SearchResult], detailed: bool = False):
        """Красивый вывод результатов"""
        if not results:
            print("\n❌ No diseases found matching the patient's test results.")
            return
        
        print("\n" + "=" * 80)
        print(f"🔍 FOUND {len(results)} POTENTIAL DISEASES")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─' * 80}")
            print(f"#{i}. {result.canonical_name} (ID: {result.disease_id})")
            print(f"{'─' * 80}")
            print(f"  📊 Match Score:       {result.total_score:.4f} / {result.max_possible_score:.4f}")
            print(f"  📈 Normalized Score:  {result.normalized_score:.2%}")
            print(f"  ✅ Matched Patterns:  {result.matched_patterns} / {result.total_patterns}")
            print(f"  ⚠️  Contradictions:    {len(result.contradictions)}")
            print(f"  ❓ Missing Data:      {len(result.missing_data)}")
            
            if detailed:
                if result.matched_details:
                    print("\n  ✅ Matched Patterns:")
                    for match in result.matched_details:
                        print(f"     • {match['test_name']}: {match['status']} "
                              f"(IDF: {match['idf_weight']:.4f}, Category: {match['category']})")
                
                if result.contradictions:
                    print("\n  ⚠️  Contradictions:")
                    for contra in result.contradictions:
                        print(f"     • {contra['test_name']}: expected {contra['expected']}, "
                              f"got {contra['actual']} (Penalty: {contra['penalty']:.4f})")
                
                if result.missing_data and len(result.missing_data) <= 5:
                    print("\n  ❓ Missing Tests (top 5):")
                    for missing in result.missing_data[:5]:
                        print(f"     • {missing['test_name']}: {missing['expected_status']} "
                              f"(IDF: {missing['idf_weight']:.4f})")
        
        print("\n" + "=" * 80)
    
    # ============================================================
    # MongoDB Integration Methods
    # ============================================================
    
    def load_references_from_mongodb(self, db_name: str = "medical_lab"):
        """
        Загрузка референсных диапазонов из MongoDB
        
        Args:
            db_name: Имя базы данных
        """
        if not self.mongodb_client:
            raise ValueError("MongoDB client not provided. Initialize with mongodb_client parameter.")
        
        print("=" * 60)
        print("Loading reference ranges from MongoDB...")
        print("=" * 60)
        
        db = self.mongodb_client[db_name]
        collection = db.reference_ranges
        
        # Получаем все документы
        documents = list(collection.find({}))
        
        if not documents:
            print("⚠️  No reference ranges found in MongoDB")
            return
        
        # Преобразуем в формат для ReferenceRangeManager
        for doc in documents:
            category = doc['test_category']
            test_name = doc['test_name']
            
            if category not in self.reference_manager.references:
                self.reference_manager.references[category] = {}
            
            # Сохраняем данные теста
            test_data = {
                'test_name': test_name,
                'alt_names': doc.get('alt_names', []),
                'units': doc.get('units', ''),
                'normal_range': doc.get('reference_ranges', {}),
                'status_ranges': doc.get('status_ranges'),
                'deviation_thresholds': doc.get('deviation_thresholds')
            }
            
            self.reference_manager.references[category][test_name] = test_data
            
            # Индексируем имена
            norm_name = self.reference_manager._normalize(test_name)
            self.reference_manager.name_index[norm_name] = (category, test_name)
            
            for alt_name in test_data['alt_names']:
                norm_alt = self.reference_manager._normalize(alt_name)
                self.reference_manager.name_index[norm_alt] = (category, test_name)
        
        total_tests = sum(len(tests) for tests in self.reference_manager.references.values())
        print(f"✓ Loaded {total_tests} reference ranges")
        print(f"✓ Built name index with {len(self.reference_manager.name_index)} entries")
        print("=" * 60)
    
    def load_diseases_from_mongodb(self, db_name: str = "medical_lab"):
        """
        Загрузка заболеваний из MongoDB
        
        Args:
            db_name: Имя базы данных
        """
        if not self.mongodb_client:
            raise ValueError("MongoDB client not provided. Initialize with mongodb_client parameter.")
        
        print("=" * 60)
        print("Loading disease database from MongoDB...")
        print("=" * 60)
        
        db = self.mongodb_client[db_name]
        collection = db.diseases
        weights_collection = db.lab_pattern_idf_weights
        pattern_weight_docs = list(weights_collection.find({}))
        pattern_weights = {doc['pattern_key']: doc for doc in pattern_weight_docs}

        print(f'? Loaded {len(pattern_weight_docs)} pattern weight entries')

        
        # Получаем все документы
        documents = list(collection.find({}))
        
        if not documents:
            print("⚠️  No diseases found in MongoDB")
            return
        
        # Инициализируем поисковый движок
        self.search_engine = DiseaseSearchEngine(self.reference_manager)
        
        # Загружаем заболевания
        for doc in documents:
            disease = Disease(
                disease_id=doc['disease_id'],
                canonical_name=doc['canonical_name'],
                patterns=[]
            )
            
            # Загружаем паттерны
            max_idf_weight = 0.0
            for pattern_data in doc.get('patterns', []):
                pattern = Pattern(
                    test_name=pattern_data['test_name'],
                    expected_status=pattern_data['expected_status'],
                    category=pattern_data['category']
                )
                pattern_key = self.search_engine._make_pattern_key(
                    pattern.test_name,
                    pattern.expected_status
                )
                weight_doc = pattern_weights.get(pattern_key)
                pattern.idf_weight = weight_doc.get('idf_weight', 1.0) if weight_doc else 1.0
                max_idf_weight += pattern.idf_weight
                disease.patterns.append(pattern)
            
            # Устанавливаем максимальный скор
            disease.max_idf_score = round(max_idf_weight, 6)
            
            # Сохраняем и индексируем
            self.search_engine.diseases[disease.disease_id] = disease
            self.search_engine._index_disease(disease)
        
        print(f"✓ Loaded {len(self.search_engine.diseases)} diseases")
        print(f"✓ Built inverted index with {len(self.search_engine.pattern_index)} patterns")
        
        # Показываем метаданные IDF
        metadata = db.metadata.find_one({"data_type": "idf_weights"})
        if metadata:
            print(f"\n📊 IDF Metadata:")
            print(f"  • Total diseases: {metadata['total_diseases']}")
            print(f"  • Unique patterns: {metadata['total_patterns']}")
            print(f"  • Avg IDF weight: {metadata['avg_idf_weight']:.4f}")
        
        print("=" * 60)
    
    def load_all_from_mongodb(self, db_name: str = "medical_lab"):
        """
        Загрузка всех данных из MongoDB
        
        Args:
            db_name: Имя базы данных
        """
        self.load_references_from_mongodb(db_name)
        self.load_diseases_from_mongodb(db_name)
    
    def get_mongodb_version(self, db_name: str = "medical_lab") -> dict:
        """
        Получение версии данных в MongoDB
        
        Returns:
            dict с информацией о версии
        """
        if not self.mongodb_client:
            raise ValueError("MongoDB client not provided")
        
        db = self.mongodb_client[db_name]
        metadata = db.metadata.find_one({"data_type": "idf_weights"})
        
        if metadata:
            return {
                "version": metadata['version'],
                "last_updated": metadata['last_updated'],
                "total_diseases": metadata['total_diseases'],
                "total_patterns": metadata['total_patterns']
            }
        
        return None
