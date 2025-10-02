#!/usr/bin/env python3
"""
index_medical_docs.py - РЎРєСЂРёРїС‚ РґР»СЏ РёРЅРґРµРєСЃР°С†РёРё РјРµРґРёС†РёРЅСЃРєРёС… JSON РґРѕРєСѓРјРµРЅС‚РѕРІ

РСЃРїРѕР»СЊР·РѕРІР°РЅРёРµ:
    python index_medical_docs.py /path/to/json/folder
    python index_medical_docs.py /path/to/json/folder --recreate
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from medical_indexer import (
    MedicalEmbedder,
    MedicalQdrantStore,
    index_medical_documents,
)


def find_json_files(folder_path: Path) -> List[Path]:
    """РџРѕРёСЃРє РІСЃРµС… JSON С„Р°Р№Р»РѕРІ РІ РїР°РїРєРµ."""
    if not folder_path.exists():
        raise FileNotFoundError(f"РџР°РїРєР° РЅРµ РЅР°Р№РґРµРЅР°: {folder_path}")

    json_files = list(folder_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"JSON С„Р°Р№Р»С‹ РЅРµ РЅР°Р№РґРµРЅС‹ РІ РїР°РїРєРµ: {folder_path}")

    return json_files


def validate_json_structure(file_path: Path) -> Dict[str, Any]:
    """Р’Р°Р»РёРґР°С†РёСЏ СЃС‚СЂСѓРєС‚СѓСЂС‹ JSON С„Р°Р№Р»Р°."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # РџСЂРѕРІРµСЂРєР° РѕР±СЏР·Р°С‚РµР»СЊРЅС‹С… РїРѕР»РµР№
        required_fields = ['doc_title', 'mkb', 'sections']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(f"РћС‚СЃСѓС‚СЃС‚РІСѓСЋС‚ РѕР±СЏР·Р°С‚РµР»СЊРЅС‹Рµ РїРѕР»СЏ: {missing_fields}")

        # РџСЂРѕРІРµСЂРєР° СЃС‚СЂСѓРєС‚СѓСЂС‹ sections
        if not isinstance(data['sections'], list):
            raise ValueError("РџРѕР»Рµ 'sections' РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ СЃРїРёСЃРєРѕРј")

        valid_sections = 0
        for i, section in enumerate(data['sections']):
            if not isinstance(section, dict):
                print(f"Р’РќРРњРђРќРР•: РЎРµРєС†РёСЏ {i} РЅРµ СЏРІР»СЏРµС‚СЃСЏ РѕР±СЉРµРєС‚РѕРј РІ {file_path.name}")
                continue

            if 'id' not in section or 'title' not in section or 'body' not in section:
                print(f"Р’РќРРњРђРќРР•: РЎРµРєС†РёСЏ {i} РЅРµ СЃРѕРґРµСЂР¶РёС‚ РѕР±СЏР·Р°С‚РµР»СЊРЅС‹С… РїРѕР»РµР№ РІ {file_path.name}")
                continue

            if section.get('body', '').strip():
                valid_sections += 1

        print(f"вњ“ {file_path.name}: {len(data['sections'])} СЃРµРєС†РёР№, {valid_sections} СЃ РєРѕРЅС‚РµРЅС‚РѕРј")
        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"РћС€РёР±РєР° РїР°СЂСЃРёРЅРіР° JSON: {e}")
    except Exception as e:
        raise ValueError(f"РћС€РёР±РєР° РІР°Р»РёРґР°С†РёРё: {e}")


def analyze_documents(json_files: List[Path]) -> Dict[str, Any]:
    """РђРЅР°Р»РёР· РІСЃРµС… РґРѕРєСѓРјРµРЅС‚РѕРІ РїРµСЂРµРґ РёРЅРґРµРєСЃР°С†РёРµР№."""

    print("=" * 60)
    print("РђРќРђР›РР— Р”РћРљРЈРњР•РќРўРћР’")
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

            # РњРљР‘ РєРѕРґС‹
            for code in data.get('mkb', []):
                icd_codes.add(code)

            # РЎРµРєС†РёРё
            doc_sections = len(data['sections'])
            doc_content_sections = sum(1 for s in data['sections'] if s.get('body', '').strip())

            total_sections += doc_sections
            total_content_sections += doc_content_sections

        except Exception as e:
            print(f"вќЊ РћС€РёР±РєР° РІ {file_path.name}: {e}")
            continue

    print(f"\nРЎРІРѕРґРєР°:")
    print(f"Р’СЃРµРіРѕ РґРѕРєСѓРјРµРЅС‚РѕРІ: {total_docs}")
    print(f"Р’СЃРµРіРѕ СЃРµРєС†РёР№: {total_sections}")
    print(f"РЎРµРєС†РёР№ СЃ РєРѕРЅС‚РµРЅС‚РѕРј: {total_content_sections}")
    print(f"РЈРЅРёРєР°Р»СЊРЅС‹С… РњРљР‘ РєРѕРґРѕРІ: {len(icd_codes)}")

    if diseases:
        print(f"\nРџРµСЂРІС‹Рµ 5 Р·Р°Р±РѕР»РµРІР°РЅРёР№:")
        for disease in diseases[:5]:
            print(f"  вЂў {disease}")
        if len(diseases) > 5:
            print(f"  ... Рё РµС‰Рµ {len(diseases) - 5}")

    return {
        'total_docs': total_docs,
        'total_sections': total_sections,
        'total_content_sections': total_content_sections,
        'icd_codes': len(icd_codes),
        'diseases': diseases
    }


def main():
    parser = argparse.ArgumentParser(description='РРЅРґРµРєСЃР°С†РёСЏ РјРµРґРёС†РёРЅСЃРєРёС… JSON РґРѕРєСѓРјРµРЅС‚РѕРІ')
    parser.add_argument('folder', help='РџСѓС‚СЊ Рє РїР°РїРєРµ СЃ JSON С„Р°Р№Р»Р°РјРё')
    parser.add_argument('--recreate', action='store_true',
                        help='РџРµСЂРµСЃРѕР·РґР°С‚СЊ РєРѕР»Р»РµРєС†РёРё (СѓРґР°Р»РёС‚ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РёРµ РґР°РЅРЅС‹Рµ)')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='URL Qdrant СЃРµСЂРІРµСЂР°')
    parser.add_argument('--model', default='intfloat/multilingual-e5-large',
                        help='РќР°Р·РІР°РЅРёРµ РјРѕРґРµР»Рё РґР»СЏ СЌРјР±РµРґРґРёРЅРіРѕРІ')
    parser.add_argument('--dry-run', action='store_true',
                        help='РўРѕР»СЊРєРѕ Р°РЅР°Р»РёР· Р±РµР· РёРЅРґРµРєСЃР°С†РёРё')

    args = parser.parse_args()

    print("рџЏҐ РРЅРґРµРєСЃР°С†РёСЏ РјРµРґРёС†РёРЅСЃРєРёС… РґРѕРєСѓРјРµРЅС‚РѕРІ")
    print(f"РџР°РїРєР°: {args.folder}")
    print(f"Qdrant: {args.qdrant_url}")
    print(f"РњРѕРґРµР»СЊ: {args.model}")
    if args.recreate:
        print("вљ пёЏ  Р РµР¶РёРј РїРµСЂРµСЃРѕР·РґР°РЅРёСЏ РєРѕР»Р»РµРєС†РёР№!")
    print()

    try:
        # РџРѕРёСЃРє JSON С„Р°Р№Р»РѕРІ
        folder_path = Path(args.folder)
        json_files = find_json_files(folder_path)
        print(f"РќР°Р№РґРµРЅРѕ JSON С„Р°Р№Р»РѕРІ: {len(json_files)}")

        # РђРЅР°Р»РёР· РґРѕРєСѓРјРµРЅС‚РѕРІ
        analysis = analyze_documents(json_files)

        if args.dry_run:
            print("\nрџ“Љ Р РµР¶РёРј dry-run. РРЅРґРµРєСЃР°С†РёСЏ РЅРµ РІС‹РїРѕР»РЅРµРЅР°.")
            return

        # РџРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ
        if args.recreate:
            confirm = input("\nРџРµСЂРµСЃРѕР·РґР°С‚СЊ РєРѕР»Р»РµРєС†РёРё? Р’СЃРµ РґР°РЅРЅС‹Рµ Р±СѓРґСѓС‚ СѓРґР°Р»РµРЅС‹! (yes/no): ")
            if confirm.lower() != 'yes':
                print("РћС‚РјРµРЅРµРЅРѕ.")
                return

        print("\n" + "=" * 60)
        print("РРќР”Р•РљРЎРђР¦РРЇ")
        print("=" * 60)

        # РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ
        print("РџРѕРґРєР»СЋС‡РµРЅРёРµ Рє Qdrant...")
        store = MedicalQdrantStore(url=args.qdrant_url)

        if not store.ping():
            print(f"вќЊ РќРµ СѓРґР°РµС‚СЃСЏ РїРѕРґРєР»СЋС‡РёС‚СЊСЃСЏ Рє Qdrant: {args.qdrant_url}")
            print("РЈР±РµРґРёС‚РµСЃСЊ С‡С‚Рѕ СЃРµСЂРІРµСЂ Р·Р°РїСѓС‰РµРЅ: docker run -p 6333:6333 qdrant/qdrant")
            return

        print("Р—Р°РіСЂСѓР·РєР° РјРѕРґРµР»Рё СЌРјР±РµРґРґРёРЅРіРѕРІ...")
        embedder = MedicalEmbedder(args.model)
        print(f"вњ“ Р Р°Р·РјРµСЂРЅРѕСЃС‚СЊ РІРµРєС‚РѕСЂРѕРІ: {embedder.get_vector_size()}")

        # РРЅРґРµРєСЃР°С†РёСЏ
        print(f"\nРќР°С‡РёРЅР°РµРј РёРЅРґРµРєСЃР°С†РёСЋ {len(json_files)} РґРѕРєСѓРјРµРЅС‚РѕРІ...")
        results = index_medical_documents(
            store,
            embedder,
            json_files,
            recreate_collections=args.recreate
        )

        print("\n" + "=" * 60)
        print("Р Р•Р—РЈР›Р¬РўРђРўР«")
        print("=" * 60)

        if 'error' in results:
            print(f"вќЊ РћС€РёР±РєР°: {results['error']}")
            return

        # РЎС‚Р°С‚РёСЃС‚РёРєР° РїРѕ РєРѕР»Р»РµРєС†РёСЏРј
        for collection_type in ['registry', 'overview', 'sections']:
            if collection_type in results:
                collection_info = results[collection_type]
                print(f"вњ“ {collection_info['collection']}: {collection_info['indexed']} Р·Р°РїРёСЃРµР№")

        # РћР±С‰Р°СЏ СЃС‚Р°С‚РёСЃС‚РёРєР°
        summary = results.get('summary', {})
        print(f"\nР’СЃРµРіРѕ:")
        print(f"  Р”РѕРєСѓРјРµРЅС‚РѕРІ: {summary.get('total_documents', 0)}")
        print(f"  Р’РµРєС‚РѕСЂРѕРІ: {summary.get('total_vectors', 0)}")
        print(f"  РљРѕР»Р»РµРєС†РёР№: {summary.get('collections_created', 0)}")

        # РРЅС„РѕСЂРјР°С†РёСЏ Рѕ РєРѕР»Р»РµРєС†РёСЏС…
        collections_info = store.get_collections_info()
        print(f"\nРС‚РѕРіРѕРІС‹Р№ СЂР°Р·РјРµСЂ РєРѕР»Р»РµРєС†РёР№:")
        for collection, count in collections_info.items():
            print(f"  {collection}: {count} С‚РѕС‡РµРє")

        print(f"\nрџЋ‰ РРЅРґРµРєСЃР°С†РёСЏ Р·Р°РІРµСЂС€РµРЅР° СѓСЃРїРµС€РЅРѕ!")
        print(f"\nРўРµРїРµСЂСЊ РјРѕР¶РЅРѕ:")
        print(f"  1. Р—Р°РїСѓСЃС‚РёС‚СЊ MCP СЃРµСЂРІРµСЂ: python medical_mcp_server.py")
        print(
            f"  2. РўРµСЃС‚РёСЂРѕРІР°С‚СЊ РїРѕРёСЃРє: python -c \"from usage_example import demo_medical_workflow; import asyncio; asyncio.run(demo_medical_workflow())\"")

    except KeyboardInterrupt:
        print("\n\nвќЊ РџСЂРµСЂРІР°РЅРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»РµРј")
    except Exception as e:
        print(f"\nвќЊ РљСЂРёС‚РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР°: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
