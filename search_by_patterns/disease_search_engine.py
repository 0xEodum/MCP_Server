"""
Medical Lab Disease Search Engine
Поисковый движок для определения заболеваний по лабораторным анализам
Version 2.0 - Improved continuous scoring with bell-shaped distribution
"""

import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import re

from search_by_patterns.models import *


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

    def distance_from_normal(
        self,
        test_name: str,
        value: float,
        gender: str = "unisex",
        units: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Возвращает:
          dist (float) ∈ [0, +∞): непрерывная дистанция от границы нормы.
          actual_status (str): one of {"normal","below_normal","above_normal","critically_low","critically_high","unknown"}.

        Определения:
          - Если value в норме → dist = 0.
          - Если выше max → dist = (value - max) / max.
          - Если ниже min → dist = (min - value) / min.
          - Если нет референсов → dist = 0, status="unknown".

        Примечание:
          Статус берём из уже существующей логики calculate_status, чтобы не дублировать правила.
        """
        status = self.calculate_status(test_name, value, gender, units)

        # Найдём референсные границы, чтобы посчитать dist
        found = self.find_test(test_name)
        if not found:
            return 0.0, "unknown"

        _category, test_data = found
        # берём гендерно-специфичный диапазон если задан
        ref = None
        if isinstance(test_data.get("normal_range"), dict):
            # ВАЖНО: сначала проверяем, не является ли это уже готовым диапазоном
            if 'min' in test_data["normal_range"] and 'max' in test_data["normal_range"]:
                # Это диапазон без разделения по полу (unisex)
                ref = test_data["normal_range"]
            else:
                # Это структура с разделением по полу
                ref = test_data["normal_range"].get(gender) or test_data["normal_range"].get("unisex")
                # Если это вложенная структура с полом
                if ref and not ('min' in ref and 'max' in ref):
                    # Попробуем взять unisex или первый доступный
                    ref = test_data["normal_range"].get("unisex")
                    if not ref or not ('min' in ref and 'max' in ref):
                        for key, rng in test_data["normal_range"].items():
                            if isinstance(rng, dict) and 'min' in rng and 'max' in rng:
                                ref = rng
                                break
        elif isinstance(test_data.get("normal_range"), (list, tuple)):
            # если где-то хранятся списком (на всякий)
            ref = test_data["normal_range"][0] if test_data["normal_range"] else None

        if not ref:
            return 0.0, status

        # попытка нормализации единиц
        unit_in = (units or "").strip().lower()
        unit_ref = (test_data.get("units") or "").strip().lower()
        v = value
        try:
            if unit_in and unit_ref and unit_in != unit_ref:
                v = self.unit_converter.convert(value, unit_in, unit_ref)
        except Exception:
            # если не смогли конвертировать — считаем как есть
            v = value

        try:
            ref_min = float(ref.get("min")) if ref.get("min") is not None else None
            ref_max = float(ref.get("max")) if ref.get("max") is not None else None
        except Exception:
            return 0.0, status

        # если нет границ — окей, дистанс 0
        if ref_min is None and ref_max is None:
            return 0.0, status

        # посчитаем dist относительно "ближней" границы
        if ref_max is not None and v > ref_max:
            base = ref_max if ref_max != 0 else abs(v) or 1.0
            dist = max(0.0, (v - ref_max) / base)
        elif ref_min is not None and v < ref_min:
            base = ref_min if ref_min != 0 else abs(v) or 1.0
            dist = max(0.0, (ref_min - v) / base)
        else:
            dist = 0.0

        return float(dist), status

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


SCORING = ScoringConfig()

def distance_contribution(
    value: float,
    status: str,
    test_data: dict,
    gender: str,
    D0: Optional[float] = None,
    max_saturation: Optional[float] = None
) -> float:
    """
    Вычисляет вклад [0..1] в зависимости от положения внутри статусного диапазона.

    Логика:
    - Для above_normal/below_normal: колоколообразная функция с максимумом в центре диапазона
      (1 - deviation_from_center²), где deviation_from_center ∈ [0, 1]
    - Для critically_high/critically_low: экспоненциальное насыщение с ограничением
      (1 - exp(-dist/D0)), ограниченное max_saturation
    - Для normal: всегда 0 (нет бонуса за "хорошую норму")

    Args:
        value: значение теста пациента
        status: статус теста ('normal', 'above_normal', 'below_normal', 'critically_high', 'critically_low')
        test_data: данные теста из reference_ranges
        gender: пол пациента
        D0: параметр насыщения для экспоненты (по умолчанию из SCORING)
        max_saturation: максимальное значение насыщения (по умолчанию из SCORING)

    Returns:
        float в диапазоне [0, max_saturation] для критических или [0, 1] для остальных
    """
    if D0 is None:
        D0 = SCORING.D0
    if max_saturation is None:
        max_saturation = SCORING.MAX_SATURATION

    if status == 'normal':
        return 0.0

    # Получаем границы статуса
    status_ranges = test_data.get('status_ranges', {})
    if status_ranges:
        ranges = status_ranges.get(gender) or status_ranges.get('unisex') or next(iter(status_ranges.values()))
        status_range = ranges.get(status) if isinstance(ranges, dict) else None
    else:
        status_range = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Для critically_high/critically_low - экспоненциальное насыщение
    # ═══════════════════════════════════════════════════════════════════════════
    if status in ('critically_high', 'critically_low'):
        if not status_range:
            return 0.5  # fallback

        if status == 'critically_high':
            boundary = status_range.get('min')
            if boundary is None:
                return 0.5
            # Дистанция за границу критической зоны
            if boundary == 0:
                dist = 0.0
            else:
                dist = max(0.0, (value - boundary) / abs(boundary))
            # Насыщение с ограничением
            contribution = 1.0 - math.exp(-dist / D0)
            return min(max_saturation, contribution)

        else:  # critically_low
            boundary = status_range.get('max')
            if boundary is None:
                return 0.5
            if boundary == 0:
                dist = 0.0
            else:
                dist = max(0.0, (boundary - value) / abs(boundary))
            contribution = 1.0 - math.exp(-dist / D0)
            return min(max_saturation, contribution)

    # ═══════════════════════════════════════════════════════════════════════════
    # Для above_normal/below_normal - колокол с максимумом в центре
    # ═══════════════════════════════════════════════════════════════════════════
    if status in ('above_normal', 'below_normal'):
        if not status_range or 'min' not in status_range or 'max' not in status_range:
            # Fallback на старую логику (экспоненциальное насыщение от границы нормы)
            normal_range = test_data.get('normal_range', {})
            if isinstance(normal_range, dict):
                # Определяем референсный диапазон
                if 'min' in normal_range and 'max' in normal_range:
                    ref = normal_range
                else:
                    ref = normal_range.get(gender) or normal_range.get('unisex')

                if ref and 'min' in ref and 'max' in ref:
                    if status == 'above_normal':
                        boundary = ref.get('max')
                    else:  # below_normal
                        boundary = ref.get('min')

                    if boundary is not None and boundary != 0:
                        dist = abs(value - boundary) / abs(boundary)
                        contribution = 1.0 - math.exp(-dist / D0)
                        return min(max_saturation, contribution)
            return 0.5  # полный fallback

        range_min = status_range['min']
        range_max = status_range['max']
        range_center = (range_min + range_max) / 2.0
        range_width = range_max - range_min

        if range_width == 0:
            return 0.5

        # Нормализованное отклонение от центра: [0..1]
        # 0 = в центре, 1 = на границе интервала
        deviation_from_center = abs(value - range_center) / (range_width / 2.0)
        deviation_from_center = min(1.0, max(0.0, deviation_from_center))

        # Колоколообразная функция: 1 - x²
        # В центре = 1.0, на краях = 0.0
        contribution = 1.0 - (deviation_from_center ** 2)
        return max(0.0, contribution)

    return 0.5  # fallback для неизвестных статусов


# ── Функции тяжести и нелинейности ─────────────────────────────────────────────
def severity_multiplier(expected: str, actual: str) -> float:
    """
    Возвращает множитель >0 при согласии направлений, <0 при противоречии,
    небольшой отрицательный при «ожидали норму» → вышли за неё.
    """
    norm = "normal"
    # нормализуем ключи
    e = (expected or "").lower()
    a = (actual or "").lower()

    # карта направлений в условные «знаки»
    def sign(s: str) -> int:
        if s in ("above_normal", "critically_high"):
            return +1
        if s in ("below_normal", "critically_low"):
            return -1
        if s in ("normal",):
            return 0
        return 0

    # «класс тяжести»
    def sev(s: str) -> int:
        if s in ("critically_high", "critically_low"):
            return 2
        if s in ("above_normal", "below_normal"):
            return 1
        if s in ("normal",):
            return 0
        return 0

    se = sev(e)
    sa = sev(a)
    sig_e = sign(e)
    sig_a = sign(a)

    if e == norm and a == norm:
        return +0.0  # ни бонуса, ни штрафа
    if e == norm and a != norm:
        # ожидали норму, а вышли — маленький минус, растёт с тяжестью факта
        return -0.2 if sa == 1 else -0.5
    if e != norm and a == norm:
        # ожидали отклонение, а его нет — умеренный минус
        return -0.6 if se == 1 else -1.0

    # оба не-норма
    if sig_e == sig_a:
        # совпало направление: плюс, сильнее для critical
        return +1.0 if (se == 1 and sa == 1) else +1.5
    else:
        # противоположные направления: заметный минус
        return -1.0 if (se == 1 and sa == 1) else -1.5


def continuous_contribution(
    idf_weight: float,
    mult: float,
    value: float,
    status: str,
    test_data: dict,
    gender: str
) -> float:
    """
    Вклад (положительный), когда ожидаемое совпало по направлению.
    mult — выход severity_multiplier (ожидаем >0).

    Гибридная формула с улучшенной логикой:
    - Базовый вклад 70% IDF за совпадение статуса
    - Бонус 30% IDF × distance_contribution за положение в интервале

    Это даёт:
    - Для above_normal/below_normal: максимум в центре диапазона
    - Для critically_high/low: экспоненциальный рост с насыщением
    - Минимум 70% IDF при совпадении статуса
    - До 100% IDF при оптимальном положении/больших отклонениях
    """
    if idf_weight is None or idf_weight <= 0:
        return 0.0

    if mult <= 0:
        return 0.0

    # Базовый вклад за совпадение статуса (70% от IDF)
    base_contribution = 0.7 * float(idf_weight) * float(mult)

    # Бонус за положение в интервале (до 30% от IDF)
    contrib = distance_contribution(value, status, test_data, gender)
    distance_bonus = 0.3 * float(idf_weight) * float(mult) * contrib

    return base_contribution + distance_bonus


def distance_penalty_contribution(
    value: float,
    status: str,
    test_data: dict,
    gender: str,
    D0: Optional[float] = None,
    max_saturation: Optional[float] = None
) -> float:
    """
    Вычисляет штрафной вклад [0..max_saturation] для неожиданных отклонений.

    Логика:
    - Для ВСЕХ статусов (above_normal, below_normal, critically_high, critically_low):
      экспоненциальное насыщение от границы нормы
    - Чем дальше от нормы, тем больше штраф
    - Для normal: 0 (нормальные значения не штрафуем)

    Это отличается от distance_contribution тем, что:
    - distance_contribution: колокол для above/below_normal (максимум в центре)
    - distance_penalty_contribution: экспонента для всех (чем дальше, тем хуже)

    Args:
        value: значение теста пациента
        status: статус теста
        test_data: данные теста из reference_ranges
        gender: пол пациента
        D0: параметр насыщения
        max_saturation: максимальное значение насыщения

    Returns:
        float в диапазоне [0, max_saturation]
    """
    if D0 is None:
        D0 = SCORING.D0
    if max_saturation is None:
        max_saturation = SCORING.MAX_SATURATION

    if status == 'normal':
        return 0.0

    # Для всех отклонений используем расстояние от границы нормы
    # Получаем нормальный диапазон
    normal_range = test_data.get('normal_range', {})

    ref = None
    if isinstance(normal_range, dict):
        if 'min' in normal_range and 'max' in normal_range:
            ref = normal_range
        else:
            ref = normal_range.get(gender) or normal_range.get('unisex')
            if not ref or not ('min' in ref and 'max' in ref):
                for key, rng in normal_range.items():
                    if isinstance(rng, dict) and 'min' in rng and 'max' in rng:
                        ref = rng
                        break

    if not ref or 'min' not in ref or 'max' not in ref:
        return 0.5  # fallback

    ref_min = ref['min']
    ref_max = ref['max']

    # Вычисляем дистанцию от границы нормы
    dist = 0.0

    if status in ('above_normal', 'critically_high'):
        # Выше нормы - считаем от верхней границы
        if ref_max == 0:
            dist = 0.0
        else:
            dist = max(0.0, (value - ref_max) / abs(ref_max))
    elif status in ('below_normal', 'critically_low'):
        # Ниже нормы - считаем от нижней границы
        if ref_min == 0:
            dist = 0.0
        else:
            dist = max(0.0, (ref_min - value) / abs(ref_min))

    # Экспоненциальное насыщение с ограничением
    contribution = 1.0 - math.exp(-dist / D0)
    return min(max_saturation, contribution)


def continuous_penalty(
    idf_weight: float,
    mult: float,
    value: float,
    status: str,
    test_data: dict,
    gender: str
) -> float:
    """
    Штраф (положительное число), когда есть противоречие или «не в паттерне».
    mult — |severity_multiplier| либо небольшой минус в случаях с нормой.

    Использует distance_penalty_contribution для корректного расчета штрафа
    в зависимости от величины отклонения.
    """
    if idf_weight is None or idf_weight <= 0:
        return 0.0

    contrib = distance_penalty_contribution(value, status, test_data, gender)
    return abs(float(idf_weight) * float(mult) * contrib)


def disease_categories_from_patterns(patterns: List) -> set:
    """
    Извлекает множество категорий, в которых у болезни заданы паттерны.
    """
    cats = set()
    for p in patterns:
        cat = getattr(p, "category", None) or (p.get("category") if isinstance(p, dict) else None)
        if cat:
            cats.add(str(cat))
    return cats


def aggregate_unexpected_penalties(
    unexpected_entries: List[Dict],
    max_m: Optional[int] = None,
    use_rms: bool = True
) -> float:
    """
    unexpected_entries: список словарей вида
        {"penalty": float, "test_name": str, ...} (penalty >= 0)
    Берём top-M по penalty, затем либо суммируем RMS, либо обычную сумму.
    """
    if not unexpected_entries:
        return 0.0
    if max_m is None:
        max_m = SCORING.TOP_M_UNEXPECTED

    top = sorted(unexpected_entries, key=lambda x: x.get("penalty", 0.0), reverse=True)[:max_m]
    vals = [max(0.0, float(x.get("penalty", 0.0))) for x in top]

    if use_rms:
        return math.sqrt(sum(v * v for v in vals))
    return sum(vals)


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

        # IDF registry для неожиданных паттернов
        self.idf_registry: Dict[str, float] = {}

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

            # Заполняем IDF registry
            for pattern in disease.patterns:
                test_info = self.reference_manager.find_test(pattern.test_name)
                if test_info:
                    _, test_data = test_info
                    canonical_name = test_data['test_name']
                else:
                    canonical_name = pattern.test_name

                pattern_key = self._make_pattern_key(canonical_name, pattern.expected_status)
                if pattern_key not in self.idf_registry:
                    self.idf_registry[pattern_key] = pattern.idf_weight

        print(f"✓ Loaded {len(self.diseases)} diseases into search engine")
        print(f"✓ Built IDF registry with {len(self.idf_registry)} patterns")

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
        Поиск заболеваний по результатам анализов пациента с улучшенным непрерывным скорингом

        Args:
            patient_tests: Список результатов анализов
            top_k: Количество результатов
            gender: Пол пациента ('male', 'female', 'unisex')
            categories: Фильтр по категориям анализов
            min_matched_patterns: Минимум совпадений для включения в результаты
            apply_contradiction_penalty: Применять ли штраф за противоречия

        Returns:
            Список результатов, отсортированный по релевантности
        """
        # 1. Определяем статусы для всех тестов пациента
        patient_patterns: Dict[str, Dict] = {}

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
                test_data = {}

            normalized_name = self._normalize(canonical_name)
            patient_patterns[normalized_name] = {
                'status': status,
                'category': category,
                'canonical_name': canonical_name,
                'normalized_name': normalized_name,
                'value': test.value,
                'units': test.units,
                'original_name': test.name,
                'test_data': test_data  # Сохраняем для distance_contribution
            }
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
            'missing_data': [],
            'redundant_data': [],
            'expected_patterns': []
        })

        # Проходим по паттернам пациента
        for normalized_name, patient_info in patient_patterns.items():
            status = patient_info['status']
            category = patient_info['category']
            canonical_name = patient_info['canonical_name']
            test_data = patient_info['test_data']

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

                    # Базовое совпадение с использованием новой логики
                    mult = 1.0  # полное совпадение
                    gain = continuous_contribution(
                        idf_weight,
                        mult,
                        patient_info['value'],
                        status,
                        test_data,
                        gender
                    )

                    # Вычисляем вклад для детализации
                    contrib = distance_contribution(
                        patient_info['value'],
                        status,
                        test_data,
                        gender
                    )

                    disease_scores[disease_id]['matched_score'] += gain
                    disease_scores[disease_id]['matched_patterns'].append({
                        'test_name': canonical_name,
                        'status': status,
                        'idf_weight': idf_weight,
                        'category': pattern_category,
                        'contribution': contrib,
                        'gain': gain
                    })

        # 3. Проверяем противоречия и missing data (только для найденных заболеваний)
        for disease_id in disease_scores.keys():
            disease = self.diseases[disease_id]

            # Получаем категории болезни
            disease_cats = disease_categories_from_patterns(disease.patterns)

            expected_entries = []
            pattern_name_map = {}
            unexpected_penalties = []

            for pattern in disease.patterns:
                # Находим каноническое название
                test_info = self.reference_manager.find_test(pattern.test_name)
                if test_info:
                    _, test_data = test_info
                    canonical_name = test_data['test_name']
                else:
                    canonical_name = pattern.test_name
                    test_data = {}

                normalized_test = self._normalize(canonical_name)

                pattern_name_map[normalized_test] = {
                    'expected_status': pattern.expected_status,
                    'idf_weight': pattern.idf_weight,
                    'category': pattern.category,
                    'canonical_name': canonical_name,
                    'test_data': test_data
                }

                expected_entries.append({
                    'test_name': canonical_name,
                    'expected_status': pattern.expected_status,
                    'category': pattern.category,
                    'idf_weight': pattern.idf_weight
                })

                # Есть ли этот тест у пациента?
                if normalized_test in patient_patterns:
                    patient_info = patient_patterns[normalized_test]
                    patient_status = patient_info['status']
                    expected_status = pattern.expected_status

                    # Если паттерн уже совпал (найден через индекс), пропускаем
                    if patient_status == expected_status:
                        continue

                    # Статусы НЕ совпадают - это всегда штраф
                    # Вопрос только в силе штрафа
                    mult = severity_multiplier(expected_status, patient_status)

                    if apply_contradiction_penalty:
                        if mult > 0:
                            # Направление совпадает, но тяжесть разная
                            # (expected: above_normal, actual: critically_high)
                            # Это меньший штраф - 40% от полного
                            penalty_multiplier = 0.4
                        else:
                            # Направление противоположное или ожидалась норма
                            # Полный штраф
                            penalty_multiplier = 1.0

                        pen = continuous_penalty(
                            pattern.idf_weight,
                            penalty_multiplier,
                            patient_info['value'],
                            patient_status,
                            test_data,
                            gender
                        )
                        disease_scores[disease_id]['contradiction_penalty'] += pen

                        # Вычисляем dist для отображения
                        dist, _ = self.reference_manager.distance_from_normal(
                            canonical_name,
                            patient_info['value'],
                            gender,
                            patient_info['units']
                        )

                        disease_scores[disease_id]['contradictions'].append({
                            'test_name': canonical_name,
                            'expected': expected_status,
                            'actual': patient_status,
                            'penalty': pen,
                            'category': pattern.category,
                            'reason': 'severity_mismatch' if mult > 0 else 'direction_mismatch',
                            'user_value': patient_info['value'],
                            'units': patient_info['units'] if patient_info['units'] else '',
                            'dist': dist
                        })
                else:
                    # Отсутствующие данные
                    disease_scores[disease_id]['missing_data'].append({
                        'test_name': canonical_name,
                        'expected_status': pattern.expected_status,
                        'reason': 'not_in_panel',
                        'idf_weight': pattern.idf_weight,
                        'category': pattern.category
                    })

            disease_scores[disease_id]['expected_patterns'] = expected_entries

            # Обработка неожиданных отклонений (не в паттерне болезни)
            pattern_name_set = set(pattern_name_map.keys())

            # Считаем default penalty
            pattern_weights = [entry['idf_weight'] for entry in expected_entries if entry.get('idf_weight') is not None]
            default_penalty = (sum(pattern_weights) / len(pattern_weights)) if pattern_weights else 0.0

            for normalized_name, patient_info in patient_patterns.items():
                # Пропускаем тесты, которые уже в паттерне
                if normalized_name in pattern_name_set:
                    continue

                # Фильтруем по категориям болезни
                if patient_info['category'] not in disease_cats:
                    continue

                actual_status = patient_info['status']
                canonical_name = patient_info['canonical_name']
                user_value = patient_info['value']
                units = patient_info['units'] if patient_info['units'] else ''
                category = patient_info['category']

                if actual_status == 'normal':
                    # Нормальные значения не штрафуем
                    disease_scores[disease_id]['redundant_data'].append({
                        'test_name': canonical_name,
                        'actual_status': actual_status,
                        'user_value': user_value,
                        'units': units,
                        'reason': 'normal_outside_pattern'
                    })
                    continue

                # Вычисляем непрерывную дистанцию (для отображения)
                dist, _ = self.reference_manager.distance_from_normal(
                    canonical_name,
                    user_value,
                    gender,
                    units
                )

                # Пытаемся взять IDF из registry
                pattern_key = self._make_pattern_key(canonical_name, actual_status)
                idf_for_unexpected = self.idf_registry.get(pattern_key, default_penalty)

                # Получаем test_data для penalty calculation
                test_info = self.reference_manager.find_test(canonical_name)
                if test_info:
                    _, test_data = test_info
                else:
                    test_data = {}

                # Множитель: ожидали норму, получили отклонение
                mult = severity_multiplier('normal', actual_status)
                pen = continuous_penalty(
                    idf_for_unexpected,
                    mult,
                    user_value,
                    actual_status,
                    test_data,
                    gender
                )

                unexpected_penalties.append({
                    'test_name': canonical_name,
                    'actual_status': actual_status,
                    'user_value': user_value,
                    'units': units,
                    'category': category,
                    'dist': dist,
                    'idf_weight': idf_for_unexpected,
                    'penalty': pen,
                    'reason': 'not_in_pattern_same_category'
                })

            # Агрегируем неожиданные штрафы
            if unexpected_penalties and apply_contradiction_penalty:
                agg_unexp = aggregate_unexpected_penalties(
                    unexpected_penalties,
                    max_m=SCORING.TOP_M_UNEXPECTED,
                    use_rms=SCORING.USE_RMS_FOR_UNEXPECTED
                )
                if agg_unexp > 0:
                    disease_scores[disease_id]['contradiction_penalty'] += agg_unexp
                    # Добавляем top-M в contradictions для объяснения
                    top_unexpected = sorted(
                        unexpected_penalties,
                        key=lambda x: x['penalty'],
                        reverse=True
                    )[:SCORING.TOP_M_UNEXPECTED]

                    for unexp in top_unexpected:
                        disease_scores[disease_id]['contradictions'].append({
                            'test_name': unexp['test_name'],
                            'expected': 'normal',
                            'actual': unexp['actual_status'],
                            'penalty': unexp['penalty'],
                            'category': unexp['category'],
                            'reason': unexp['reason'],
                            'user_value': unexp['user_value'],
                            'units': unexp['units'],
                            'dist': unexp.get('dist', 0.0)
                        })

        # 4. Финальный скоринг и формирование результатов
        results = []

        for disease_id, scores in disease_scores.items():
            if len(scores['matched_patterns']) < min_matched_patterns:
                continue

            disease = self.diseases[disease_id]

            matched_score = scores['matched_score']
            contradiction_penalty = scores['contradiction_penalty']

            # Применяем cap на штрафы
            max_score = disease.max_idf_score
            if max_score > 0:
                cap = SCORING.CONTRADICTION_CAP_FRAC * max_score
                contradiction_penalty = min(contradiction_penalty, cap)

            total_score = matched_score - contradiction_penalty

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
                missing_data=scores['missing_data'],
                redundant_data=scores['redundant_data'],
                expected_patterns=scores['expected_patterns']
            )

            results.append(result)

        # 5. Сортировка по релевантности
        results.sort(
            key=lambda x: (x.total_score, x.normalized_score, x.matched_patterns),
            reverse=True
        )

        filtered_results = [result for result in results if result.total_score >= 0]

        return filtered_results[:top_k]

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

    def explain_tests(
        self,
        tests: List[Dict],
        gender: str = 'unisex'
    ) -> List[Dict]:
        '''Build explanations for raw user test data.'''
        explanations = []
        for test in tests:
            name = str(test.get('name', '') or '').strip()
            raw_value = test.get('value')
            input_units = test.get('units', '') or ''
            numeric_value = self._safe_float(raw_value)

            entry = {
                'test_name': {
                    'value': name or None,
                    'units': input_units or None
                },
                'user_value': {
                    'value': numeric_value if numeric_value is not None else raw_value,
                    'units': input_units or None
                },
                'reference_value': {
                    'value': None,
                    'units': None
                },
                'status': {
                    'value': 'unknown'
                }
            }

            if not name:
                explanations.append(entry)
                continue

            test_info = self.reference_manager.find_test(name)
            if not test_info:
                explanations.append(entry)
                continue

            _, test_data = test_info
            canonical_name = test_data.get('test_name', name)
            target_units = test_data.get('units') or input_units or ''
            display_units = target_units or input_units or ''

            entry['test_name']['value'] = canonical_name
            entry['test_name']['units'] = display_units or None

            if numeric_value is not None:
                converted_value = self.reference_manager.unit_converter.convert(
                    numeric_value,
                    input_units,
                    target_units
                )
                entry['user_value']['value'] = converted_value
                entry['user_value']['units'] = display_units or None
            else:
                entry['user_value']['units'] = display_units or None

            range_data, _ = self._extract_normal_range(test_data, gender)
            if range_data:
                entry['reference_value']['value'] = {
                    'min': range_data.get('min'),
                    'max': range_data.get('max')
                }
                entry['reference_value']['units'] = display_units or None

            status_value = 'unknown'
            if numeric_value is not None:
                status_value = self.reference_manager.calculate_status(
                    canonical_name,
                    numeric_value,
                    gender=gender,
                    units=input_units
                )
            entry['status']['value'] = status_value

            for field_key in ('test_name', 'user_value', 'reference_value'):
                if entry[field_key].get('units') is None:
                    entry[field_key].pop('units', None)

            explanations.append(entry)

        return explanations

    @staticmethod
    def _safe_float(raw_value) -> Optional[float]:
        if raw_value is None:
            return None
        try:
            return float(str(raw_value).replace(',', '.'))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_range_dict(data) -> bool:
        return isinstance(data, dict) and 'min' in data and 'max' in data

    @staticmethod
    def _extract_normal_range(test_data: Dict, gender: str) -> Tuple[Optional[Dict], Optional[str]]:
        normal_range = test_data.get('normal_range')
        if isinstance(normal_range, dict):
            if gender in normal_range and MedicalLabAnalyzer._is_range_dict(normal_range[gender]):
                return normal_range[gender], gender
            if 'unisex' in normal_range and MedicalLabAnalyzer._is_range_dict(normal_range['unisex']):
                return normal_range['unisex'], 'unisex'
            if MedicalLabAnalyzer._is_range_dict(normal_range):
                return normal_range, 'unisex'
            for key, rng in normal_range.items():
                if MedicalLabAnalyzer._is_range_dict(rng):
                    return rng, key
        status_ranges = test_data.get('status_ranges')
        if isinstance(status_ranges, dict):
            if gender in status_ranges:
                gender_ranges = status_ranges[gender]
                if isinstance(gender_ranges, dict):
                    normal = gender_ranges.get('normal')
                    if MedicalLabAnalyzer._is_range_dict(normal):
                        return normal, gender
            if 'unisex' in status_ranges:
                gender_ranges = status_ranges['unisex']
                if isinstance(gender_ranges, dict):
                    normal = gender_ranges.get('normal')
                    if MedicalLabAnalyzer._is_range_dict(normal):
                        return normal, 'unisex'
            for key, gender_ranges in status_ranges.items():
                if isinstance(gender_ranges, dict):
                    normal = gender_ranges.get('normal')
                    if MedicalLabAnalyzer._is_range_dict(normal):
                        return normal, key
        return None, None

    @staticmethod
    def _format_number(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.6g}"

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

        print(f'✓ Loaded {len(pattern_weight_docs)} pattern weight entries')

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

            # Заполняем IDF registry
            for pattern in disease.patterns:
                test_info = self.reference_manager.find_test(pattern.test_name)
                if test_info:
                    _, test_data = test_info
                    canonical_name = test_data['test_name']
                else:
                    canonical_name = pattern.test_name

                pattern_key = self.search_engine._make_pattern_key(canonical_name, pattern.expected_status)
                if pattern_key not in self.search_engine.idf_registry:
                    self.search_engine.idf_registry[pattern_key] = pattern.idf_weight

        print(f"✓ Loaded {len(self.search_engine.diseases)} diseases")
        print(f"✓ Built inverted index with {len(self.search_engine.pattern_index)} patterns")
        print(f"✓ Built IDF registry with {len(self.search_engine.idf_registry)} patterns")

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