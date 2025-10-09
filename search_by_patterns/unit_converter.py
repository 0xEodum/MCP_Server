"""
Unit Converter for Medical Laboratory Values
Конвертер единиц измерения для лабораторных показателей

Поддерживает гибкое преобразование между различными форматами единиц измерения
с использованием regex для парсинга сложных единиц.
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class UnitComponents:
    """Компоненты распарсенной единицы измерения"""
    base_unit: str  # основная единица (g, mol, etc.)
    power: Optional[int] = None  # степень (9, 12 для 10^9, 10^12)
    denominator: str = ''  # знаменатель (l, dl, etc.)
    is_percentage: bool = False
    
    def to_canonical(self) -> str:
        """Приведение к каноническому виду"""
        if self.is_percentage:
            return 'percent'
        
        if not self.base_unit:
            return ''
        
        result = self.base_unit
        
        if self.power is not None:
            result = f'1e{self.power}'
        elif self.denominator:
            result = f'{self.base_unit}_per_{self.denominator}'
        
        return result


class UnitConverter:
    """
    Конвертер единиц измерения с поддержкой regex-парсинга
    
    Примеры поддерживаемых форматов:
    - x10^9/л, 10^9/l, ×10⁹/л
    - x10^12/л, 10^12/l
    - g/l, g/dl, мг/дл
    - mmol/l, µmol/l, ммоль/л
    - %, percent
    """
    
    def __init__(self):
        self._init_replacements()
        self._init_patterns()
        self._init_base_conversions()
    
    def _init_replacements(self):
        """Инициализация таблицы замен для нормализации"""
        self._REPLACEMENTS = [
            # Символы умножения
            ('×', 'x'),
            ('х', 'x'),  # русская х
            ('*', 'x'),
            # Дефисы и минусы
            ('−', '-'),
            ('–', '-'),
            # Микро-символы
            ('µl', 'ul'),
            ('μl', 'ul'),
            ('µ', 'u'),
            ('μ', 'u'),
            # Русские единицы → латиница
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
            # Запятая → точка для чисел
            (',', '.'),
        ]
    
    def _init_patterns(self):
        """Инициализация regex-паттернов для парсинга"""
        # Паттерн для степеней: x10^9, 10^12, ×10⁹, etc.
        self._power_pattern = re.compile(
            r'(?:x|×)?\s*10\s*[\^⁰¹²³⁴⁵⁶⁷⁸⁹]\s*(\d+)',
            re.IGNORECASE
        )
        
        # Паттерн для единиц с дробью: g/l, mg/dl, mmol/l
        self._fraction_pattern = re.compile(
            r'([a-z]+)\s*/\s*([a-z]+)',
            re.IGNORECASE
        )
        
        # Паттерн для процентов
        self._percent_pattern = re.compile(r'%|percent', re.IGNORECASE)
        
        # Паттерн для простых единиц
        self._simple_unit_pattern = re.compile(r'^([a-z]+)$', re.IGNORECASE)
    
    def _init_base_conversions(self):
        """Инициализация базовых коэффициентов конвертации"""
        # Базовые конвертации между префиксами СИ
        self._SI_PREFIXES = {
            'g': 1.0,      # грамм (базовая единица массы)
            'mg': 0.001,   # миллиграмм = 0.001 г
            'ug': 1e-6,    # микрограмм = 0.000001 г
            'kg': 1000.0,  # килограмм = 1000 г
        }
        
        # Конвертации объёма
        self._VOLUME_UNITS = {
            'l': 1.0,      # литр (базовая единица объёма)
            'dl': 0.1,     # децилитр = 0.1 л
            'ml': 0.001,   # миллилитр = 0.001 л
            'ul': 1e-6,    # микролитр = 0.000001 л
        }
        
        # Конвертации молярности
        self._MOLAR_UNITS = {
            'mol': 1.0,    # моль (базовая единица)
            'mmol': 0.001, # миллимоль = 0.001 моль
            'umol': 1e-6,  # микромоль = 0.000001 моль
        }
        
        # Конвертации степеней больше не нужны!
        # Используем универсальную математическую формулу:
        # value × 10^a → value × 10^b
        # factor = 10^(a - b)

    def normalize_input(self, unit: Optional[str]) -> str:
        """
        Первичная нормализация строки единицы

        Args:
            unit: Исходная строка единицы

        Returns:
            Нормализованная строка
        """
        if not unit:
            return ''

        # Приводим к нижнему регистру и убираем лишние пробелы
        normalized = unit.strip().lower()

        if not normalized:
            return ''

        # Применяем все замены
        for src, dst in self._REPLACEMENTS:
            normalized = normalized.replace(src, dst)

        # Убираем множественные пробелы
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    def parse_unit(self, unit_str: str) -> UnitComponents:
        """
        Парсинг единицы измерения в компоненты

        Args:
            unit_str: Нормализованная строка единицы

        Returns:
            UnitComponents с распарсенными компонентами
        """
        if not unit_str:
            return UnitComponents(base_unit='')

        # Проверяем процент
        if self._percent_pattern.search(unit_str):
            return UnitComponents(base_unit='', is_percentage=True)

        # Ищем степень (10^9, 10^12, etc.)
        power_match = self._power_pattern.search(unit_str)
        if power_match:
            power = int(power_match.group(1))

            # Ищем знаменатель после степени
            remaining = unit_str[power_match.end():]
            denominator = ''

            if '/' in remaining:
                denom_match = re.search(r'/\s*([a-z]+)', remaining, re.IGNORECASE)
                if denom_match:
                    denominator = denom_match.group(1)

            return UnitComponents(
                base_unit='cell_count',  # специальная метка для клеточных единиц
                power=power,
                denominator=denominator
            )

        # Ищем дробные единицы (g/l, mg/dl, etc.)
        fraction_match = self._fraction_pattern.search(unit_str)
        if fraction_match:
            numerator = fraction_match.group(1)
            denominator = fraction_match.group(2)

            return UnitComponents(
                base_unit=numerator,
                denominator=denominator
            )

        # Простая единица (unit, pg, fl, etc.)
        simple_match = self._simple_unit_pattern.match(unit_str)
        if simple_match:
            return UnitComponents(base_unit=simple_match.group(1))

        # Не смогли распарсить
        return UnitComponents(base_unit=unit_str)

    def get_conversion_factor(
        self,
        from_components: UnitComponents,
        to_components: UnitComponents
    ) -> Optional[float]:
        """
        Вычисление коэффициента конвертации между единицами

        Args:
            from_components: Компоненты исходной единицы
            to_components: Компоненты целевой единицы

        Returns:
            Коэффициент конвертации или None если конвертация невозможна
        """
        # Проценты конвертируются только в проценты
        if from_components.is_percentage and to_components.is_percentage:
            return 1.0
        if from_components.is_percentage or to_components.is_percentage:
            return None

        # Одинаковые единицы
        if from_components == to_components:
            return 1.0

        # ===============================================
        # Конвертация степеней (клеточные единицы)
        # ===============================================
        if (from_components.base_unit == 'cell_count' and
            to_components.base_unit == 'cell_count'):

            from_power = from_components.power or 0
            to_power = to_components.power or 0

            # Проверяем совместимость знаменателей
            if from_components.denominator != to_components.denominator:
                return None

            # Универсальная формула для любых степеней:
            # value × 10^a → value × 10^b
            # factor = 10^(a - b)
            #
            # Примеры:
            # - 35 × 10^11 → ? × 10^9:  factor = 10^(11-9) = 100  → 3500
            # - 3.5 × 10^12 → ? × 10^9: factor = 10^(12-9) = 1000 → 3500
            # - 3500 × 10^9 → ? × 10^12: factor = 10^(9-12) = 0.001 → 3.5

            power_diff = from_power - to_power
            factor = 10 ** power_diff

            return factor

        # ===============================================
        # Конвертация дробных единиц (концентрации)
        # ===============================================
        if from_components.denominator and to_components.denominator:
            # Должны быть совместимые типы
            from_num = from_components.base_unit
            to_num = to_components.base_unit

            # Проверяем совместимость числителей
            from_num_type = self._get_unit_type(from_num)
            to_num_type = self._get_unit_type(to_num)

            if from_num_type != to_num_type or from_num_type is None:
                return None

            # Получаем факторы для числителя и знаменателя
            num_factor = self._get_unit_factor(from_num, to_num, from_num_type)
            denom_factor = self._get_unit_factor(
                from_components.denominator,
                to_components.denominator,
                'volume'
            )

            if num_factor is None or denom_factor is None:
                return None

            # g/l -> g/dl: числитель не меняется (1), знаменатель 0.1
            # итого: 1 / 0.1 = 10 (умножаем на 10) ✓
            return num_factor / denom_factor

        # ===============================================
        # Простые единицы (unit, pg, fl)
        # ===============================================
        if (not from_components.denominator and
            not to_components.denominator and
            not from_components.power and
            not to_components.power):

            # Прямая конвертация для одинаковых единиц
            if from_components.base_unit == to_components.base_unit:
                return 1.0

        # Конвертация невозможна
        return None

    def _get_unit_type(self, unit: str) -> Optional[str]:
        """Определение типа единицы (масса, объём, молярность)"""
        if unit in self._SI_PREFIXES:
            return 'mass'
        if unit in self._VOLUME_UNITS:
            return 'volume'
        if unit in self._MOLAR_UNITS:
            return 'molar'
        return None

    def _get_unit_factor(
        self,
        from_unit: str,
        to_unit: str,
        unit_type: str
    ) -> Optional[float]:
        """Получение фактора конвертации для единиц одного типа"""
        lookup = {
            'mass': self._SI_PREFIXES,
            'volume': self._VOLUME_UNITS,
            'molar': self._MOLAR_UNITS
        }

        unit_dict = lookup.get(unit_type)
        if not unit_dict:
            return None

        from_factor = unit_dict.get(from_unit)
        to_factor = unit_dict.get(to_unit)

        if from_factor is None or to_factor is None:
            return None

        return from_factor / to_factor

    def convert(
        self,
        value: float,
        from_unit: Optional[str],
        to_unit: Optional[str]
    ) -> float:
        """
        Конвертация значения между единицами измерения

        Args:
            value: Исходное значение
            from_unit: Исходная единица (может быть None)
            to_unit: Целевая единица (может быть None)

        Returns:
            Сконвертированное значение

        Examples:
            >>> converter = UnitConverter()
            >>> converter.convert(0.141, "x10^12/л", "x10^9/л")
            141.0
            >>> converter.convert(10.5, "g/l", "g/dl")
            1.05
            >>> converter.convert(150, "mmol/l", "umol/l")
            150000.0
        """
        # Если единицы не указаны или одинаковые, возвращаем как есть
        if not from_unit or not to_unit:
            return value

        # Нормализуем входные единицы
        from_normalized = self.normalize_input(from_unit)
        to_normalized = self.normalize_input(to_unit)

        if not from_normalized or not to_normalized:
            return value

        if from_normalized == to_normalized:
            return value

        # Парсим единицы
        from_components = self.parse_unit(from_normalized)
        to_components = self.parse_unit(to_normalized)

        # Получаем коэффициент конвертации
        factor = self.get_conversion_factor(from_components, to_components)

        if factor is None:
            # Конвертация невозможна, возвращаем исходное значение
            return value

        # Применяем конвертацию
        return value * factor

    def are_compatible(
        self,
        unit1: Optional[str],
        unit2: Optional[str]
    ) -> bool:
        """
        Проверка совместимости единиц измерения

        Args:
            unit1: Первая единица
            unit2: Вторая единица

        Returns:
            True если единицы совместимы для конвертации
        """
        if not unit1 or not unit2:
            return True  # Пустые единицы считаем совместимыми

        norm1 = self.normalize_input(unit1)
        norm2 = self.normalize_input(unit2)

        if norm1 == norm2:
            return True

        comp1 = self.parse_unit(norm1)
        comp2 = self.parse_unit(norm2)

        factor = self.get_conversion_factor(comp1, comp2)
        return factor is not None


# ============================================================
# Вспомогательные функции для удобства использования
# ============================================================

def convert_unit(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Удобная функция для быстрой конвертации

    Args:
        value: Значение для конвертации
        from_unit: Исходная единица
        to_unit: Целевая единица

    Returns:
        Сконвертированное значение
    """
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit)


def are_units_compatible(unit1: str, unit2: str) -> bool:
    """
    Проверка совместимости единиц

    Args:
        unit1: Первая единица
        unit2: Вторая единица

    Returns:
        True если единицы совместимы
    """
    converter = UnitConverter()
    return converter.are_compatible(unit1, unit2)