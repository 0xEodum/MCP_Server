#!/usr/bin/env python3
"""
index_medical_docs.py - Скрипт для индексации медицинских JSON документов

Использование:
    python index_medical_docs.py /path/to/json/folder
    python index_medical_docs.py /path/to/json/folder --recreate
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from medical_qdrant_api import MedicalQdrantStore
from medical_embedding_api import MedicalEmbedder
from medical_indexer import index_medical_documents


def find_json_files(folder_path: Path) -> List[Path]:
    """Поиск всех JSON файлов в папке."""
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")

    json_files = list(folder_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"JSON файлы не найдены в папке: {folder_path}")

    return json_files


def validate_json_structure(file_path: Path) -> Dict[str, Any]:
    """Валидация структуры JSON файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверка обязательных полей
        required_fields = ['doc_title', 'mkb', 'sections']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(f"Отсутствуют обязательные поля: {missing_fields}")

        # Проверка структуры sections
        if not isinstance(data['sections'], list):
            raise ValueError("Поле 'sections' должно быть списком")

        valid_sections = 0
        for i, section in enumerate(data['sections']):
            if not isinstance(section, dict):
                print(f"ВНИМАНИЕ: Секция {i} не является объектом в {file_path.name}")
                continue

            if 'id' not in section or 'title' not in section or 'body' not in section:
                print(f"ВНИМАНИЕ: Секция {i} не содержит обязательных полей в {file_path.name}")
                continue

            if section.get('body', '').strip():
                valid_sections += 1

        print(f"✓ {file_path.name}: {len(data['sections'])} секций, {valid_sections} с контентом")
        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON: {e}")
    except Exception as e:
        raise ValueError(f"Ошибка валидации: {e}")


def analyze_documents(json_files: List[Path]) -> Dict[str, Any]:
    """Анализ всех документов перед индексацией."""

    print("=" * 60)
    print("АНАЛИЗ ДОКУМЕНТОВ")
    print("=" * 60)

    total_docs = 0
    total_sections = 0
    total_content_sections = 0
    diseases = []
    icd_codes = set()

    for file_path in json_files:
        try:
            data = validate_json_structure(file_path)

            total_docs += 1
            diseases.append(data['doc_title'])

            # МКБ коды
            for code in data.get('mkb', []):
                icd_codes.add(code)

            # Секции
            doc_sections = len(data['sections'])
            doc_content_sections = sum(1 for s in data['sections'] if s.get('body', '').strip())

            total_sections += doc_sections
            total_content_sections += doc_content_sections

        except Exception as e:
            print(f"❌ Ошибка в {file_path.name}: {e}")
            continue

    print(f"\nСводка:")
    print(f"Всего документов: {total_docs}")
    print(f"Всего секций: {total_sections}")
    print(f"Секций с контентом: {total_content_sections}")
    print(f"Уникальных МКБ кодов: {len(icd_codes)}")

    if diseases:
        print(f"\nПервые 5 заболеваний:")
        for disease in diseases[:5]:
            print(f"  • {disease}")
        if len(diseases) > 5:
            print(f"  ... и еще {len(diseases) - 5}")

    return {
        'total_docs': total_docs,
        'total_sections': total_sections,
        'total_content_sections': total_content_sections,
        'icd_codes': len(icd_codes),
        'diseases': diseases
    }


def main():
    parser = argparse.ArgumentParser(description='Индексация медицинских JSON документов')
    parser.add_argument('folder', help='Путь к папке с JSON файлами')
    parser.add_argument('--recreate', action='store_true',
                        help='Пересоздать коллекции (удалит существующие данные)')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='URL Qdrant сервера')
    parser.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Название модели для эмбеддингов')
    parser.add_argument('--dry-run', action='store_true',
                        help='Только анализ без индексации')

    args = parser.parse_args()

    print("🏥 Индексация медицинских документов")
    print(f"Папка: {args.folder}")
    print(f"Qdrant: {args.qdrant_url}")
    print(f"Модель: {args.model}")
    if args.recreate:
        print("⚠️  Режим пересоздания коллекций!")
    print()

    try:
        # Поиск JSON файлов
        folder_path = Path(args.folder)
        json_files = find_json_files(folder_path)
        print(f"Найдено JSON файлов: {len(json_files)}")

        # Анализ документов
        analysis = analyze_documents(json_files)

        if args.dry_run:
            print("\n📊 Режим dry-run. Индексация не выполнена.")
            return

        # Подтверждение
        if args.recreate:
            confirm = input("\nПересоздать коллекции? Все данные будут удалены! (yes/no): ")
            if confirm.lower() != 'yes':
                print("Отменено.")
                return

        print("\n" + "=" * 60)
        print("ИНДЕКСАЦИЯ")
        print("=" * 60)

        # Инициализация
        print("Подключение к Qdrant...")
        store = MedicalQdrantStore(url=args.qdrant_url)

        if not store.ping():
            print(f"❌ Не удается подключиться к Qdrant: {args.qdrant_url}")
            print("Убедитесь что сервер запущен: docker run -p 6333:6333 qdrant/qdrant")
            return

        print("Загрузка модели эмбеддингов...")
        embedder = MedicalEmbedder(args.model)
        print(f"✓ Размерность векторов: {embedder.get_vector_size()}")

        # Индексация
        print(f"\nНачинаем индексацию {len(json_files)} документов...")
        results = index_medical_documents(
            store,
            embedder,
            json_files,
            recreate_collections=args.recreate
        )

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ")
        print("=" * 60)

        if 'error' in results:
            print(f"❌ Ошибка: {results['error']}")
            return

        # Статистика по коллекциям
        for collection_type in ['registry', 'overview', 'sections']:
            if collection_type in results:
                collection_info = results[collection_type]
                print(f"✓ {collection_info['collection']}: {collection_info['indexed']} записей")

        # Общая статистика
        summary = results.get('summary', {})
        print(f"\nВсего:")
        print(f"  Документов: {summary.get('total_documents', 0)}")
        print(f"  Векторов: {summary.get('total_vectors', 0)}")
        print(f"  Коллекций: {summary.get('collections_created', 0)}")

        # Информация о коллекциях
        collections_info = store.get_collections_info()
        print(f"\nИтоговый размер коллекций:")
        for collection, count in collections_info.items():
            print(f"  {collection}: {count} точек")

        print(f"\n🎉 Индексация завершена успешно!")
        print(f"\nТеперь можно:")
        print(f"  1. Запустить MCP сервер: python medical_mcp_server.py")
        print(
            f"  2. Тестировать поиск: python -c \"from usage_example import demo_medical_workflow; import asyncio; asyncio.run(demo_medical_workflow())\"")

    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())