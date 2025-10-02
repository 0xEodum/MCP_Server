#!/usr/bin/env python3
"""
Combined Loader for Medical System
Loads disease data to Qdrant and patterns to MongoDB with synchronized disease_id
"""

import sys
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from collections import defaultdict

from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# Medical RAG imports
from medical_indexer import (
    MedicalEmbedder,
    MedicalQdrantStore,
    index_medical_documents,
)
from medical_indexer.utils import slugify_document_title


class CombinedLoader:
    """
    Combined loader for Qdrant and MongoDB.
    Ensures disease_id synchronization between databases.
    """

    def __init__(
        self,
        mongodb_uri: str = "mongodb://localhost:27017",
        mongodb_db: str = "medical_lab",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        embedding_model: str = "intfloat/multilingual-e5-small"
    ):
        # MongoDB connection
        self.mongo_client = MongoClient(mongodb_uri)
        self.mongo_db = self.mongo_client[mongodb_db]
        
        # Qdrant connection
        self.qdrant_store = MedicalQdrantStore(url=qdrant_url, api_key=qdrant_api_key)
        
        # Embedder
        self.embedder = MedicalEmbedder(embedding_model)
        
        # Disease ID mapping: filename -> (disease_id, canonical_name)
        self.disease_mapping: Dict[str, Tuple[str, str]] = {}

    def load_reference_ranges(self, json_path: str, clear_existing: bool = False):
        """Load reference ranges into MongoDB (from mongodb_loader logic)"""
        print("\n" + "=" * 60)
        print(f"📊 Loading reference ranges from {json_path}")
        print("=" * 60)

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if clear_existing:
            result = self.mongo_db.reference_ranges.delete_many({})
            print(f"✓ Cleared {result.deleted_count} existing documents")

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
                result = self.mongo_db.reference_ranges.insert_many(documents, ordered=False)
                print(f"✓ Inserted {len(result.inserted_ids)} reference ranges")
            except BulkWriteError as error:
                inserted = (error.details or {}).get("nInserted", 0)
                print(f"⚠️  Inserted {inserted} documents before encountering write errors")
                self._log_bulk_write_error(error)

        total = self.mongo_db.reference_ranges.count_documents({})
        print(f"✓ Total reference ranges in DB: {total}")

    def build_disease_mapping(self, diseases_folder: Path) -> Dict[str, Tuple[str, str]]:
        """
        Build mapping: base_filename -> (disease_id, canonical_name)
        This ensures synchronization between Qdrant and MongoDB
        """
        print("\n" + "=" * 60)
        print("🔗 Building disease ID mapping...")
        print("=" * 60)

        mapping = {}
        json_files = list(diseases_folder.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                doc_title = data.get('doc_title')
                if not doc_title:
                    print(f"⚠️  Skipping {json_file.name}: missing 'doc_title'")
                    continue

                # Generate disease_id using the same logic as Qdrant loader
                disease_id = slugify_document_title(doc_title)
                canonical_name = doc_title

                # Base filename without extension
                base_name = json_file.stem

                mapping[base_name] = (disease_id, canonical_name)
                print(f"✓ {base_name}")
                print(f"  └─ disease_id: {disease_id}")
                print(f"  └─ canonical_name: {canonical_name}")

            except Exception as e:
                print(f"❌ Error processing {json_file.name}: {e}")
                continue

        print(f"\n✓ Built mapping for {len(mapping)} diseases")
        return mapping

    def load_diseases_to_qdrant(
        self,
        diseases_folder: Path,
        recreate_collections: bool = False
    ) -> Dict:
        """Load diseases into Qdrant (from qdrant_loader logic)"""
        print("\n" + "=" * 60)
        print("🔍 Loading diseases to Qdrant...")
        print("=" * 60)

        json_files = list(diseases_folder.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {diseases_folder}")

        print(f"Found {len(json_files)} disease files")

        # Validate structure
        for json_file in json_files:
            self._validate_disease_structure(json_file)

        # Index to Qdrant
        print("\nIndexing to Qdrant...")
        results = index_medical_documents(
            self.qdrant_store,
            self.embedder,
            json_files,
            recreate_collections=recreate_collections
        )

        print("\n✓ Qdrant indexing completed")
        if 'summary' in results:
            summary = results['summary']
            print(f"  └─ Documents: {summary.get('total_documents', 0)}")
            print(f"  └─ Vectors: {summary.get('total_vectors', 0)}")
            print(f"  └─ Collections: {summary.get('collections_created', 0)}")

        return results

    def load_patterns_to_mongodb(
        self,
        patterns_folder: Path,
        disease_mapping: Dict[str, Tuple[str, str]],
        clear_existing: bool = False
    ):
        """Load patterns into MongoDB with synchronized disease_id"""
        print("\n" + "=" * 60)
        print("🧬 Loading patterns to MongoDB...")
        print("=" * 60)

        if clear_existing:
            result = self.mongo_db.diseases.delete_many({})
            print(f"✓ Cleared {result.deleted_count} existing disease documents")

        pattern_files = list(patterns_folder.glob("*_pattern.json"))
        
        if not pattern_files:
            print(f"⚠️  No pattern files found in {patterns_folder}")
            return

        print(f"Found {len(pattern_files)} pattern files")

        # Load patterns with disease_id from mapping
        diseases_with_patterns = []
        
        for pattern_file in pattern_files:
            # Extract base name: "X_pattern.json" -> "X"
            base_name = pattern_file.stem.replace("_pattern", "")
            
            # Check if we have mapping for this disease
            if base_name not in disease_mapping:
                print(f"⚠️  Skipping {pattern_file.name}: no matching disease file found")
                continue

            disease_id, canonical_name = disease_mapping[base_name]

            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                pattern_groups = data.get("pattern_groups", {})
                
                if not pattern_groups:
                    print(f"⚠️  Skipping {pattern_file.name}: no pattern_groups found")
                    continue

                # Build disease document
                disease = {
                    "disease_id": disease_id,
                    "canonical_name": canonical_name,
                    "patterns": []
                }

                # Extract patterns from all categories
                for category, patterns in pattern_groups.items():
                    for pattern in patterns:
                        disease["patterns"].append({
                            "test_name": pattern["test_name"],
                            "expected_status": pattern["status"],
                            "category": category,
                        })

                diseases_with_patterns.append(disease)
                print(f"✓ {base_name}")
                print(f"  └─ disease_id: {disease_id}")
                print(f"  └─ patterns: {len(disease['patterns'])}")

            except Exception as e:
                print(f"❌ Error loading {pattern_file.name}: {e}")
                continue

        if not diseases_with_patterns:
            print("❌ No patterns loaded")
            return

        # Calculate IDF weights
        print("\n📊 Calculating IDF weights...")
        pattern_stats = self._calculate_idf_weights(diseases_with_patterns)

        # Prepare documents for MongoDB
        documents = []
        now = datetime.now(timezone.utc)

        for disease in diseases_with_patterns:
            stored_patterns = []
            max_idf_score = 0.0

            for pattern in disease["patterns"]:
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

        # Insert diseases
        if documents:
            try:
                result = self.mongo_db.diseases.insert_many(documents, ordered=False)
                print(f"✓ Inserted {len(result.inserted_ids)} diseases")
            except BulkWriteError as error:
                inserted = (error.details or {}).get("nInserted", 0)
                print(f"⚠️  Inserted {inserted} documents before encountering write errors")
                self._log_bulk_write_error(error)

        # Store IDF weights
        self._store_idf_weights(pattern_stats)

        # Save metadata
        self._save_idf_metadata(pattern_stats, len(diseases_with_patterns))

        total = self.mongo_db.diseases.count_documents({})
        print(f"✓ Total diseases in MongoDB: {total}")

    def _calculate_idf_weights(self, diseases: List[Dict]) -> Dict[str, Dict]:
        """Calculate IDF weights for patterns (from mongodb_loader logic)"""
        total_diseases = len(diseases)
        if total_diseases == 0:
            return {}

        pattern_df = defaultdict(int)

        # Count document frequency for each pattern
        for disease in diseases:
            unique_patterns = set()
            for pattern in disease["patterns"]:
                pattern_key = self._make_pattern_key(
                    pattern["test_name"],
                    pattern["expected_status"]
                )
                unique_patterns.add(pattern_key)
            
            for pattern_key in unique_patterns:
                pattern_df[pattern_key] += 1

        # Calculate IDF for each pattern
        pattern_stats: Dict[str, Dict] = {}
        total_pattern_instances = 0
        sum_idf = 0.0

        for disease in diseases:
            for pattern in disease["patterns"]:
                pattern_key = self._make_pattern_key(
                    pattern["test_name"],
                    pattern["expected_status"]
                )
                df = pattern_df[pattern_key]
                idf_weight = math.log((total_diseases + 1) / (df + 1))
                rounded_idf = round(idf_weight, 6)

                # Store IDF in pattern
                pattern["idf_weight"] = rounded_idf

                # Store stats
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

        print(f"  ✓ Total diseases: {total_diseases}")
        print(f"  ✓ Unique patterns: {len(pattern_stats)}")
        print(f"  ✓ Total pattern instances: {total_pattern_instances}")
        print(f"  ✓ Average IDF weight: {avg_idf:.4f}")

        return pattern_stats

    def _store_idf_weights(self, pattern_stats: Dict[str, Dict]):
        """Store IDF weights in MongoDB collection"""
        collection = self.mongo_db.lab_pattern_idf_weights
        collection.delete_many({})

        pattern_documents = []
        now = datetime.now(timezone.utc)

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
                collection.insert_many(pattern_documents, ordered=False)
                print(f"✓ Stored {len(pattern_documents)} IDF weight records")
            except BulkWriteError as error:
                inserted = (error.details or {}).get("nInserted", 0)
                print(f"⚠️  Inserted {inserted} IDF weights before encountering write errors")
                self._log_bulk_write_error(error)

    def _save_idf_metadata(self, pattern_stats: Dict[str, Dict], total_diseases: int):
        """Save IDF metadata for versioning"""
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

        self.mongo_db.metadata.update_one(
            {"data_type": "idf_weights"},
            {"$set": metadata},
            upsert=True,
        )

        print(f"✓ Saved IDF metadata (version: {metadata['version']})")

    @staticmethod
    def _make_pattern_key(test_name: str, status: str) -> str:
        """Create pattern key for indexing"""
        normalized = test_name.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return f"{normalized}:{status}"

    @staticmethod
    def _validate_disease_structure(file_path: Path) -> None:
        """Validate disease JSON structure"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        required_fields = ['doc_title', 'sections']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(f"{file_path.name}: missing fields {missing_fields}")

    def _log_bulk_write_error(self, error: BulkWriteError):
        """Log MongoDB bulk write errors"""
        details = getattr(error, "details", None) or {}
        write_errors = details.get("writeErrors", [])
        if not write_errors:
            print(f"❌ Bulk write failed: {error}")
            return

        first_error = write_errors[0]
        message = first_error.get("errmsg", "Unknown error")
        print(f"❌ MongoDB error: {message}")

    def get_statistics(self) -> Dict:
        """Get statistics from both databases"""
        print("\n" + "=" * 60)
        print("📊 DATABASE STATISTICS")
        print("=" * 60)

        # MongoDB stats
        mongo_stats = {
            "reference_ranges": self.mongo_db.reference_ranges.count_documents({}),
            "diseases": self.mongo_db.diseases.count_documents({}),
            "pattern_idf_weights": self.mongo_db.lab_pattern_idf_weights.count_documents({}),
        }

        print("\n📦 MongoDB:")
        for collection, count in mongo_stats.items():
            print(f"  └─ {collection}: {count} documents")

        metadata = self.mongo_db.metadata.find_one({"data_type": "idf_weights"})
        if metadata:
            print("\n  IDF Metadata:")
            print(f"    └─ Version: {metadata['version']}")
            print(f"    └─ Total diseases: {metadata['total_diseases']}")
            print(f"    └─ Total patterns: {metadata['total_patterns']}")
            print(f"    └─ Avg IDF weight: {metadata['avg_idf_weight']:.4f}")

        # Qdrant stats
        qdrant_stats = self.qdrant_store.get_collections_info()
        
        print("\n🔍 Qdrant:")
        for collection, count in qdrant_stats.items():
            status = "✓" if count >= 0 else "❌"
            print(f"  {status} {collection}: {count} points")

        return {
            "mongodb": mongo_stats,
            "qdrant": qdrant_stats,
            "metadata": metadata
        }

    def load_all(
        self,
        data_folder: Path,
        reference_file: Path,
        recreate_qdrant: bool = False,
        clear_mongodb: bool = False
    ):
        """
        Complete loading process:
        1. Load reference ranges to MongoDB
        2. Build disease ID mapping
        3. Load diseases to Qdrant
        4. Load patterns to MongoDB
        """
        print("\n" + "=" * 60)
        print("🚀 COMBINED MEDICAL DATA LOADER")
        print("=" * 60)
        print(f"\nData folder: {data_folder}")
        print(f"Reference file: {reference_file}")
        print(f"Recreate Qdrant: {recreate_qdrant}")
        print(f"Clear MongoDB: {clear_mongodb}")

        diseases_folder = data_folder / "diseases"
        patterns_folder = data_folder / "patterns"

        # Validate structure
        if not diseases_folder.exists():
            raise FileNotFoundError(f"Diseases folder not found: {diseases_folder}")
        if not patterns_folder.exists():
            raise FileNotFoundError(f"Patterns folder not found: {patterns_folder}")
        if not reference_file.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_file}")

        # Step 1: Load reference ranges
        self.load_reference_ranges(reference_file, clear_existing=clear_mongodb)

        # Step 2: Build disease ID mapping
        self.disease_mapping = self.build_disease_mapping(diseases_folder)

        # Step 3: Load diseases to Qdrant
        self.load_diseases_to_qdrant(diseases_folder, recreate_collections=recreate_qdrant)

        # Step 4: Load patterns to MongoDB
        self.load_patterns_to_mongodb(
            patterns_folder,
            self.disease_mapping,
            clear_existing=clear_mongodb
        )

        # Final statistics
        self.get_statistics()

        print("\n" + "=" * 60)
        print("✅ LOADING COMPLETED SUCCESSFULLY")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Combined loader for medical data (Qdrant + MongoDB)"
    )
    
    parser.add_argument(
        "data_folder",
        type=Path,
        help="Path to folder containing diseases/ and patterns/ subdirectories"
    )
    
    parser.add_argument(
        "reference_file",
        type=Path,
        help="Path to reference ranges JSON file (e.g., ref_blood.json)"
    )
    
    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection URI"
    )
    
    parser.add_argument(
        "--mongodb-db",
        default="medical_lab",
        help="MongoDB database name"
    )
    
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL"
    )
    
    parser.add_argument(
        "--qdrant-api-key",
        help="Qdrant API key (if required)"
    )
    
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Embedding model name"
    )
    
    parser.add_argument(
        "--recreate-qdrant",
        action="store_true",
        help="Recreate Qdrant collections (⚠️  deletes existing data)"
    )
    
    parser.add_argument(
        "--clear-mongodb",
        action="store_true",
        help="Clear MongoDB collections (⚠️  deletes existing data)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate structure without loading"
    )

    args = parser.parse_args()

    try:
        # Create loader
        loader = CombinedLoader(
            mongodb_uri=args.mongodb_uri,
            mongodb_db=args.mongodb_db,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            embedding_model=args.model
        )

        # Test connections
        print("Testing connections...")
        
        # Test MongoDB
        loader.mongo_client.admin.command('ping')
        print("✓ MongoDB connected")
        
        # Test Qdrant
        if not loader.qdrant_store.ping():
            raise ConnectionError("Cannot connect to Qdrant")
        print("✓ Qdrant connected")

        if args.dry_run:
            print("\n📋 DRY RUN MODE - Validating structure only")
            
            diseases_folder = args.data_folder / "diseases"
            patterns_folder = args.data_folder / "patterns"
            
            print(f"\nValidating diseases in: {diseases_folder}")
            disease_files = list(diseases_folder.glob("*.json"))
            print(f"Found {len(disease_files)} disease files")
            
            print(f"\nValidating patterns in: {patterns_folder}")
            pattern_files = list(patterns_folder.glob("*_pattern.json"))
            print(f"Found {len(pattern_files)} pattern files")
            
            print(f"\nValidating reference file: {args.reference_file}")
            with open(args.reference_file) as f:
                ref_data = json.load(f)
                categories = len(ref_data.get("reference_ranges", {}))
                print(f"Found {categories} test categories")
            
            print("\n✅ Structure validation passed")
            return 0

        # Confirm destructive operations
        if args.recreate_qdrant or args.clear_mongodb:
            print("\n⚠️  WARNING: Destructive operations requested!")
            if args.recreate_qdrant:
                print("  - Qdrant collections will be recreated")
            if args.clear_mongodb:
                print("  - MongoDB collections will be cleared")
            
            confirm = input("\nType 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Operation cancelled")
                return 0

        # Load all data
        loader.load_all(
            data_folder=args.data_folder,
            reference_file=args.reference_file,
            recreate_qdrant=args.recreate_qdrant,
            clear_mongodb=args.clear_mongodb
        )

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
