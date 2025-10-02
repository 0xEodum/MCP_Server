"""
Data Loader for Medical Lab Disease Search Engine
Loads JSON fixtures into MongoDB and computes pattern IDF weights
"""

import json
import math
import re
from datetime import datetime, timezone
from typing import Dict, List
from collections import defaultdict
from pymongo import MongoClient
from pymongo.errors import BulkWriteError


class DataLoader:
    def __init__(self, connection_string: str = "mongodb://localhost:27017", db_name: str = "medical_lab"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]

    def load_reference_ranges(self, json_path: str, clear_existing: bool = False):
        print("=" * 60)
        print(f"Loading reference ranges from {json_path}")
        print("=" * 60)

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if clear_existing:
            result = self.db.reference_ranges.delete_many({})
            print(f"[OK] Cleared {result.deleted_count} existing documents")

        documents = []
        ref_ranges = data.get("reference_ranges", {})

        for category, tests in ref_ranges.items():
            for test in tests:
                now = datetime.now(timezone.utc)
                documents.append({
                    "test_name": test["test_name"],
                    "test_category": category,
                    "alt_names": test.get("alt_names") or [],
                    "units": test.get("units") or "",
                    "reference_ranges": test.get("normal_range") or test.get("reference_ranges") or {},
                    "status_ranges": test.get("status_ranges") or {},
                    "deviation_thresholds": test.get("deviation_thresholds") or {},
                    "created_at": now,
                    "updated_at": now,
                })

        if documents:
            try:
                result = self.db.reference_ranges.insert_many(documents, ordered=False)
                print(f"[OK] Inserted {len(result.inserted_ids)} reference ranges")
            except BulkWriteError as error:
                inserted = (error.details or {}).get("nInserted", 0)
                print(f"[WARN] Inserted {inserted} documents before encountering write errors")
                self._log_bulk_write_error(error)

        total = self.db.reference_ranges.count_documents({})
        print(f"[OK] Total reference ranges in DB: {total}")

    def load_diseases(self, json_path: str, clear_existing: bool = False):
        print("\n" + "=" * 60)
        print(f"Loading diseases from {json_path}")
        print("=" * 60)

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        diseases_data = data.get("diseases", [])

        if clear_existing:
            result = self.db.diseases.delete_many({})
            print(f"[OK] Cleared {result.deleted_count} existing documents")

        diseases_with_patterns: List[Dict] = []
        for disease_data in diseases_data:
            disease = {
                "disease_id": disease_data["disease_id"],
                "canonical_name": disease_data["canonical_name"],
                "patterns": [],
            }

            pattern_groups = disease_data.get("pattern_groups", {})
            for category, patterns in pattern_groups.items():
                for pattern in patterns:
                    disease["patterns"].append({
                        "test_name": pattern["test_name"],
                        "expected_status": pattern["status"],
                        "category": category,
                    })

            diseases_with_patterns.append(disease)

        print("\n[INFO] Calculating IDF weights...")
        pattern_stats = self._calculate_idf_weights(diseases_with_patterns)

        documents = []
        now = datetime.now(timezone.utc)

        for disease in diseases_with_patterns:
            stored_patterns = []
            max_idf_score = 0.0

            for pattern in disease["patterns"]:
                pattern_key = self._make_pattern_key(pattern["test_name"], pattern["expected_status"])
                idf_weight = pattern.get("idf_weight", 1.0)
                max_idf_score += idf_weight
                stored_patterns.append({
                    "test_name": pattern["test_name"],
                    "expected_status": pattern["expected_status"],
                    "category": pattern["category"],
                })

            documents.append({
                "disease_id": disease["disease_id"],
                "canonical_name": disease["canonical_name"],
                "patterns": stored_patterns,
                "total_patterns": len(stored_patterns),
                "max_idf_score": round(max_idf_score, 6),
                "created_at": now,
                "updated_at": now,
            })

        if documents:
            try:
                result = self.db.diseases.insert_many(documents, ordered=False)
                print(f"[OK] Inserted {len(result.inserted_ids)} diseases")
            except BulkWriteError as error:
                inserted = (error.details or {}).get("nInserted", 0)
                print(f"[WARN] Inserted {inserted} documents before encountering write errors")
                self._log_bulk_write_error(error)

        pattern_collection = self.db.lab_pattern_idf_weights
        pattern_collection.delete_many({})

        pattern_documents = []
        for pattern_key, stats in pattern_stats.items():
            pattern_documents.append({
                "pattern_key": pattern_key,
                "test_name": stats["test_name"],
                "expected_status": stats["expected_status"],
                "idf_weight": stats["idf_weight"],
                "document_frequency": stats["document_frequency"],
                "total_diseases": stats["total_diseases"],
                "updated_at": now,
            })

        if pattern_documents:
            try:
                pattern_collection.insert_many(pattern_documents, ordered=False)
                print(f"[OK] Upserted {len(pattern_documents)} pattern IDF records")
            except BulkWriteError as error:
                inserted = (error.details or {}).get("nInserted", 0)
                print(f"[WARN] Inserted {inserted} pattern weights before encountering write errors")
                self._log_bulk_write_error(error)
        else:
            print("[WARN] No pattern IDF weights generated")

        total_diseases = self.db.diseases.count_documents({})
        print(f"[OK] Total diseases in DB: {total_diseases}")

        self._save_idf_metadata(pattern_stats, len(diseases_with_patterns))

    def _calculate_idf_weights(self, diseases: List[Dict]) -> Dict[str, Dict]:
        total_diseases = len(diseases)
        if total_diseases == 0:
            return {}

        pattern_df = defaultdict(int)

        for disease in diseases:
            unique_patterns = set()
            for pattern in disease["patterns"]:
                pattern_key = self._make_pattern_key(pattern["test_name"], pattern["expected_status"])
                unique_patterns.add(pattern_key)
            for pattern_key in unique_patterns:
                pattern_df[pattern_key] += 1

        pattern_stats: Dict[str, Dict] = {}
        total_pattern_instances = 0
        sum_idf = 0.0

        for disease in diseases:
            for pattern in disease["patterns"]:
                pattern_key = self._make_pattern_key(pattern["test_name"], pattern["expected_status"])
                df = pattern_df[pattern_key]
                idf_weight = math.log((total_diseases + 1) / (df + 1))
                rounded_idf = round(idf_weight, 6)

                pattern["idf_weight"] = rounded_idf
                if pattern_key not in pattern_stats:
                    pattern_stats[pattern_key] = {
                        "pattern_key": pattern_key,
                        "test_name": pattern["test_name"],
                        "expected_status": pattern["expected_status"],
                        "idf_weight": rounded_idf,
                        "document_frequency": df,
                        "total_diseases": total_diseases,
                    }

                total_pattern_instances += 1
                sum_idf += idf_weight

        avg_idf = sum_idf / total_pattern_instances if total_pattern_instances else 0.0

        print("  [OK] Calculated IDF weights")
        print(f"  - Total diseases: {total_diseases}")
        print(f"  - Unique patterns: {len(pattern_stats)}")
        print(f"  - Total pattern instances: {total_pattern_instances}")
        print(f"  - Average IDF weight: {avg_idf:.4f}")

        return pattern_stats

    def _save_idf_metadata(self, pattern_stats: Dict[str, Dict], total_diseases: int):
        total_patterns = len(pattern_stats)
        sum_idf = sum(stat["idf_weight"] for stat in pattern_stats.values())
        avg_idf = sum_idf / total_patterns if total_patterns else 0.0

        now = datetime.now(timezone.utc)
        metadata = {
            "data_type": "idf_weights",
            "version": int(now.timestamp()),
            "last_updated": now,
            "total_diseases": total_diseases,
            "total_patterns": total_patterns,
            "avg_idf_weight": round(avg_idf, 6),
        }

        self.db.metadata.update_one(
            {"data_type": "idf_weights"},
            {"$set": metadata},
            upsert=True,
        )

        print(f"\n[OK] Saved IDF metadata (version: {metadata['version']})")

    def _log_bulk_write_error(self, error: BulkWriteError):
        details = getattr(error, "details", None) or {}
        write_errors = details.get("writeErrors", [])
        if not write_errors:
            print(f"[ERROR] Bulk write failed: {error}")
            return

        first_error = write_errors[0]
        message = first_error.get("errmsg", "Unknown error")
        print(f"[ERROR] MongoDB error: {message}")

        err_info = first_error.get("errInfo", {})
        failing_fields = self._extract_validation_fields(err_info)
        if failing_fields:
            joined = ", ".join(sorted(failing_fields))
            print(f"[ERROR] Fields failing validation: {joined}")

    @staticmethod
    def _extract_validation_fields(err_info: Dict) -> List[str]:
        if not err_info:
            return []

        fields = set()
        details = err_info.get("details", {})
        for rule in details.get("schemaRulesNotSatisfied", []):
            for prop in rule.get("propertiesNotSatisfied", []):
                name = prop.get("propertyName")
                if name:
                    fields.add(name)
        return list(fields)

    @staticmethod
    def _make_pattern_key(test_name: str, status: str) -> str:
        normalized = test_name.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return f"{normalized}:{status}"

    def load_all(self, ref_json: str, diseases_json: str, clear_existing: bool = False):
        print("\n" + "=" * 58)
        print("|   Loading All Data to MongoDB                          |")
        print("=" * 58 + "\n")

        self.load_reference_ranges(ref_json, clear_existing)
        self.load_diseases(diseases_json, clear_existing)

        print("\n" + "=" * 60)
        print("[OK] DATA LOADING COMPLETED")
        print("=" * 60)
        self.get_stats()

    def get_stats(self):
        stats = {
            "reference_ranges": self.db.reference_ranges.count_documents({}),
            "diseases": self.db.diseases.count_documents({}),
            "pattern_idf_weights": self.db.lab_pattern_idf_weights.count_documents({}),
        }

        print("\n[INFO] Database Statistics:")
        for collection, count in stats.items():
            print(f"  - {collection}: {count} documents")

        metadata = self.db.metadata.find_one({"data_type": "idf_weights"})
        if metadata:
            print("\n[INFO] IDF Metadata:")
            print(f"  - Version: {metadata['version']}")
            print(f"  - Last updated: {metadata['last_updated']}")
            print(f"  - Total diseases: {metadata['total_diseases']}")
            print(f"  - Total patterns: {metadata['total_patterns']}")
            print(f"  - Avg IDF weight: {metadata['avg_idf_weight']:.4f}")

        return stats


if __name__ == "__main__":
    import sys

    print("=" * 58)
    print("|   MongoDB Data Loader                                  |")
    print("=" * 58)

    connection_string = "mongodb://localhost:27017"
    db_name = "medical_lab"
    ref_json = "ref_blood.json"
    diseases_json = "diseases.json"

    if len(sys.argv) > 1:
        ref_json = sys.argv[1]
    if len(sys.argv) > 2:
        diseases_json = sys.argv[2]

    print(f"\nConnection: {connection_string}")
    print(f"Database: {db_name}")
    print(f"Reference file: {ref_json}")
    print(f"Diseases file: {diseases_json}\n")

    loader = DataLoader(connection_string, db_name)

    print("Options:")
    print("  1) Load all data (keep existing)")
    print("  2) Load all data (clear existing)")
    print("  3) Load only reference ranges")
    print("  4) Load only diseases")
    print("  5) Show statistics")
    print("  6) Exit")

    choice = input("\nYour choice: ").strip()

    try:
        if choice == "1":
            loader.load_all(ref_json, diseases_json, clear_existing=False)
        elif choice == "2":
            confirm = input("\nWARNING: This will clear existing data. Continue? (yes/no): ")
            if confirm.lower() == "yes":
                loader.load_all(ref_json, diseases_json, clear_existing=True)
        elif choice == "3":
            clear = input("Clear existing? (yes/no): ").lower() == "yes"
            loader.load_reference_ranges(ref_json, clear_existing=clear)
        elif choice == "4":
            clear = input("Clear existing? (yes/no): ").lower() == "yes"
            loader.load_diseases(diseases_json, clear_existing=clear)
        elif choice == "5":
            loader.get_stats()
        else:
            print("Goodbye!")
    except FileNotFoundError as error:
        print(f"\n[ERROR] Error: {error}")
        print("Make sure JSON files exist in the current directory")
    except Exception as error:
        print(f"\n[ERROR] Error: {error}")
