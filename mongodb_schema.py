"""
MongoDB Schema для Medical Lab Disease Search Engine
Коллекции и индексы для оптимальной работы
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid


class MongoDBSchema:
    """Управление схемой MongoDB"""
    
    def __init__(self, connection_string="mongodb://localhost:27017", db_name="medical_lab"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
    
    def create_collections(self):
        """Создание коллекций с валидацией"""
        print("=" * 60)
        print("Creating MongoDB collections...")
        print("=" * 60)
        
        # 1. Коллекция reference_ranges
        self._create_reference_ranges_collection()
        
        # 2. ��������� diseases
        self._create_diseases_collection()
        
        # 3. ��������� lab_pattern_idf_weights
        self._create_pattern_idf_collection()
        
        # 4. ��������� metadata
        self._create_metadata_collection()
        
        print("\n✅ All collections created successfully!")
    
    def _create_reference_ranges_collection(self):
        """Коллекция референсных диапазонов"""
        collection_name = "reference_ranges"
        
        # JSON Schema для валидации
        validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["test_name", "test_category", "units", "reference_ranges"],
                "properties": {
                    "test_name": {
                        "bsonType": "string",
                        "description": "Название теста (каноническое)"
                    },
                    "test_category": {
                        "bsonType": "string",
                        "description": "Категория теста"
                    },
                    "alt_names": {
                        "bsonType": "array",
                        "items": {"bsonType": "string"},
                        "description": "Альтернативные названия"
                    },
                    "units": {
                        "bsonType": "string",
                        "description": "Единицы измерения"
                    },
                    "reference_ranges": {
                        "bsonType": "object",
                        "description": "Референсные диапазоны по полу"
                    },
                    "status_ranges": {
                        "bsonType": "object",
                        "description": "Абсолютные диапазоны статусов"
                    },
                    "deviation_thresholds": {
                        "bsonType": "object",
                        "description": "Процентные пороги отклонений"
                    },
                    "created_at": {
                        "bsonType": "date",
                        "description": "Дата создания"
                    },
                    "updated_at": {
                        "bsonType": "date",
                        "description": "Дата обновления"
                    }
                }
            }
        }
        
        try:
            self.db.create_collection(collection_name, validator=validator)
            print(f"✓ Created collection: {collection_name}")
        except CollectionInvalid:
            print(f"[WARN] Collection {collection_name} already exists")
        
        # Создание индексов
        collection = self.db[collection_name]
        
        collection.create_index([("test_name", ASCENDING)], unique=False)
        collection.create_index([("test_category", ASCENDING)])
        collection.create_index([("alt_names", ASCENDING)])
        collection.create_index(
            [("test_name", ASCENDING), ("test_category", ASCENDING)],
            unique=True,
            name="unique_test_category"
        )
        
        print(f"  ✓ Created indexes for {collection_name}")
    
    def _create_diseases_collection(self):
        """Коллекция заболеваний с паттернами"""
        collection_name = "diseases"
        
        # JSON Schema для валидации
        validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["disease_id", "canonical_name", "patterns"],
                "properties": {
                    "disease_id": {
                        "bsonType": "string",
                        "description": "Уникальный ID заболевания"
                    },
                    "canonical_name": {
                        "bsonType": "string",
                        "description": "Каноническое название"
                    },
                    "patterns": {
                        "bsonType": "array",
                        "description": "Массив паттернов",
                        "items": {
                            "bsonType": "object",
                            "required": ["test_name", "expected_status", "category"],
                            "properties": {
                                "test_name": {"bsonType": "string"},
                                "expected_status": {
                                    "bsonType": "string",
                                    "enum": ["normal", "below_normal", "above_normal", 
                                            "critically_low", "critically_high"]
                                },
                                "category": {"bsonType": "string"}
                            }
                        }
                    },
                    "total_patterns": {
                        "bsonType": "int",
                        "description": "Общее количество паттернов"
                    },
                    "max_idf_score": {
                        "bsonType": "double",
                        "description": "Максимальный возможный IDF скор"
                    },
                    "created_at": {
                        "bsonType": "date",
                        "description": "Дата создания"
                    },
                    "updated_at": {
                        "bsonType": "date",
                        "description": "Дата обновления"
                    }
                }
            }
        }
        
        try:
            self.db.create_collection(collection_name, validator=validator)
            print(f"✓ Created collection: {collection_name}")
        except CollectionInvalid:
            print(f"[WARN] Collection {collection_name} already exists")
        
        # Создание индексов
        collection = self.db[collection_name]
        
        collection.create_index([("disease_id", ASCENDING)], unique=True)
        collection.create_index([("patterns.test_name", ASCENDING)])
        collection.create_index([("patterns.expected_status", ASCENDING)])
        collection.create_index([("patterns.category", ASCENDING)])
        collection.create_index([("patterns.idf_weight", DESCENDING)])
        collection.create_index([
            ("patterns.test_name", ASCENDING),
            ("patterns.expected_status", ASCENDING)
        ])
        
        print(f"  ✓ Created indexes for {collection_name}")
    
    def _create_pattern_idf_collection(self):
        """�������� lab_pattern_idf_weights"""
        collection_name = "lab_pattern_idf_weights"

        validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["pattern_key", "test_name", "expected_status", "idf_weight", "document_frequency", "total_diseases"],
                "properties": {
                    "pattern_key": {"bsonType": "string"},
                    "test_name": {"bsonType": "string"},
                    "expected_status": {"bsonType": "string"},
                    "idf_weight": {"bsonType": "double"},
                    "document_frequency": {"bsonType": "int"},
                    "total_diseases": {"bsonType": "int"},
                    "updated_at": {"bsonType": "date"}
                }
            }
        }

        try:
            self.db.create_collection(collection_name, validator=validator)
            print(f"? Created collection: {collection_name}")
        except CollectionInvalid:
            print(f"[WARN] Collection {collection_name} already exists")

        collection = self.db[collection_name]
        collection.create_index([("pattern_key", ASCENDING)], unique=True)
        collection.create_index([("test_name", ASCENDING)])
        collection.create_index([("expected_status", ASCENDING)])

        print(f"  ? Created indexes for {collection_name}")

    def _create_metadata_collection(self):
        """Коллекция метаданных"""
        collection_name = "metadata"
        
        validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["data_type", "version", "last_updated"],
                "properties": {
                    "data_type": {
                        "bsonType": "string",
                        "enum": ["idf_weights", "system_config"],
                        "description": "Тип метаданных"
                    },
                    "version": {
                        "bsonType": "int",
                        "description": "Версия данных"
                    },
                    "last_updated": {
                        "bsonType": "date",
                        "description": "Последнее обновление"
                    },
                    "total_diseases": {
                        "bsonType": "int",
                        "description": "Количество заболеваний"
                    },
                    "total_patterns": {
                        "bsonType": "int",
                        "description": "Количество уникальных паттернов"
                    },
                    "avg_idf_weight": {
                        "bsonType": "double",
                        "description": "Средний IDF вес"
                    }
                }
            }
        }
        
        try:
            self.db.create_collection(collection_name, validator=validator)
            print(f"✓ Created collection: {collection_name}")
        except CollectionInvalid:
            print(f"[WARN] Collection {collection_name} already exists")
        
        # Индекс
        collection = self.db[collection_name]
        collection.create_index([("data_type", ASCENDING)], unique=True)
        
        print(f"  ✓ Created indexes for {collection_name}")
    
    def drop_all_collections(self):
        """Удаление всех коллекций (осторожно!)"""
        print("\n⚠️  WARNING: Dropping all collections...")
        
        for collection_name in ["reference_ranges", "diseases", "lab_pattern_idf_weights", "metadata"]:
            self.db.drop_collection(collection_name)
            print(f"  ✗ Dropped: {collection_name}")
        
        print("✓ All collections dropped")
    
    def get_stats(self):
        """Статистика базы данных"""
        print("\n" + "=" * 60)
        print("MongoDB Statistics")
        print("=" * 60)
        
        stats = {
            "reference_ranges": self.db.reference_ranges.count_documents({}),
            "diseases": self.db.diseases.count_documents({}),
            "lab_pattern_idf_weights": self.db.lab_pattern_idf_weights.count_documents({}),
            "metadata": self.db.metadata.count_documents({})
        }
        
        for collection, count in stats.items():
            print(f"  {collection}: {count} documents")
        
        # Размер базы
        db_stats = self.db.command("dbStats")
        size_mb = db_stats.get("dataSize", 0) / (1024 * 1024)
        print(f"\n  Database size: {size_mb:.2f} MB")
        
        return stats


if __name__ == "__main__":
    import sys
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║   MongoDB Schema Setup                                 ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Параметры подключения
    connection_string = "mongodb://localhost:27017"
    db_name = "medical_lab"
    
    if len(sys.argv) > 1:
        connection_string = sys.argv[1]
    if len(sys.argv) > 2:
        db_name = sys.argv[2]
    
    print(f"\nConnection: {connection_string}")
    print(f"Database: {db_name}\n")
    
    schema = MongoDBSchema(connection_string, db_name)
    
    # Меню
    print("\nOptions:")
    print("  1) Create collections and indexes")
    print("  2) Drop all collections (⚠️  DANGER!)")
    print("  3) Show statistics")
    print("  4) Exit")
    
    choice = input("\nYour choice: ").strip()
    
    if choice == "1":
        schema.create_collections()
        schema.get_stats()
    elif choice == "2":
        confirm = input("\n⚠️  Are you sure? Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            schema.drop_all_collections()
    elif choice == "3":
        schema.get_stats()
    else:
        print("Goodbye!")

