"""
Sync Manager для Medical Lab Disease Search Engine
Управление синхронизацией между MongoDB и in-memory поисковым движком
"""

import asyncio
import time
from datetime import datetime
from typing import Optional
from pymongo import MongoClient

from search_by_patterns.disease_search_engine import MedicalLabAnalyzer


class SyncManager:
    """
    Менеджер синхронизации данных
    Отслеживает изменения в MongoDB и обновляет in-memory движок
    """
    
    def __init__(
        self,
        analyzer: MedicalLabAnalyzer,
        mongodb_client: MongoClient,
        db_name: str = "medical_lab",
        check_interval: int = 3600  # 1 час по умолчанию
    ):
        """
        Args:
            analyzer: Экземпляр MedicalLabAnalyzer
            mongodb_client: MongoClient для подключения к MongoDB
            db_name: Имя базы данных
            check_interval: Интервал проверки обновлений (секунды)
        """
        self.analyzer = analyzer
        self.mongodb_client = mongodb_client
        self.db_name = db_name
        self.check_interval = check_interval
        self.db = mongodb_client[db_name]
        
        self.current_version: Optional[int] = None
        self.last_check_time: Optional[datetime] = None
        self.is_running = False
    
    async def start(self):
        """Запуск фонового процесса синхронизации"""
        self.is_running = True
        
        print("=" * 60)
        print("🔄 Sync Manager Started")
        print("=" * 60)
        print(f"  Database: {self.db_name}")
        print(f"  Check interval: {self.check_interval} seconds")
        print("=" * 60)
        
        # Первоначальная загрузка
        await self._check_and_sync()
        
        # Основной цикл
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_and_sync()
            except Exception as e:
                print(f"❌ Sync error: {e}")
                await asyncio.sleep(60)  # Подождать минуту при ошибке
    
    def stop(self):
        """Остановка синхронизации"""
        self.is_running = False
        print("\n🛑 Sync Manager Stopped")
    
    async def _check_and_sync(self):
        """Проверка версии и синхронизация при необходимости"""
        self.last_check_time = datetime.utcnow()
        
        # Получаем текущую версию из MongoDB
        metadata = self.db.metadata.find_one({"data_type": "idf_weights"})
        
        if not metadata:
            print("\n⚠️  No metadata found in MongoDB")
            return
        
        db_version = metadata['version']
        
        # Первая загрузка или версия изменилась?
        if self.current_version is None:
            print("\n📥 Initial data load from MongoDB...")
            await self._sync_data()
            self.current_version = db_version
        elif db_version != self.current_version:
            print(f"\n📥 Data version changed: {self.current_version} → {db_version}")
            await self._sync_data()
            self.current_version = db_version
        else:
            print(f"\n✓ Data is up-to-date (version: {self.current_version})")
    
    async def _sync_data(self):
        """Синхронизация данных из MongoDB"""
        start_time = time.time()
        
        try:
            # Загружаем данные из MongoDB
            self.analyzer.load_references_from_mongodb(self.db_name)
            self.analyzer.load_diseases_from_mongodb(self.db_name)
            
            elapsed = (time.time() - start_time) * 1000
            print(f"✅ Data synchronized in {elapsed:.2f} ms")
            
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            raise
    
    def force_sync(self):
        """Принудительная синхронизация (для вызова из API)"""
        print("\n🔄 Force sync requested...")
        
        start_time = time.time()
        
        self.analyzer.load_references_from_mongodb(self.db_name)
        self.analyzer.load_diseases_from_mongodb(self.db_name)
        
        # Обновляем версию
        metadata = self.db.metadata.find_one({"data_type": "idf_weights"})
        if metadata:
            self.current_version = metadata['version']
        
        elapsed = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "version": self.current_version,
            "sync_time_ms": elapsed,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_status(self) -> dict:
        """Получение статуса синхронизации"""
        metadata = self.db.metadata.find_one({"data_type": "idf_weights"})
        
        db_version = metadata['version'] if metadata else None
        in_sync = db_version == self.current_version if db_version else False
        
        return {
            "is_running": self.is_running,
            "current_version": self.current_version,
            "db_version": db_version,
            "in_sync": in_sync,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "check_interval_seconds": self.check_interval
        }


class SimpleSyncManager:
    """
    Упрощённый менеджер синхронизации без asyncio
    Для использования в простых приложениях
    """
    
    def __init__(
        self,
        analyzer: MedicalLabAnalyzer,
        mongodb_client: MongoClient,
        db_name: str = "medical_lab"
    ):
        self.analyzer = analyzer
        self.mongodb_client = mongodb_client
        self.db_name = db_name
        self.db = mongodb_client[db_name]
        self.current_version: Optional[int] = None
    
    def sync_if_changed(self) -> bool:
        """
        Проверка и синхронизация если данные изменились
        
        Returns:
            True если данные обновились, False если нет изменений
        """
        metadata = self.db.metadata.find_one({"data_type": "idf_weights"})
        
        if not metadata:
            return False
        
        db_version = metadata['version']
        
        # Первая загрузка или версия изменилась?
        if self.current_version is None or db_version != self.current_version:
            print(f"\n📥 Syncing data from MongoDB (version: {db_version})...")
            
            start_time = time.time()
            
            self.analyzer.load_references_from_mongodb(self.db_name)
            self.analyzer.load_diseases_from_mongodb(self.db_name)
            
            self.current_version = db_version
            
            elapsed = (time.time() - start_time) * 1000
            print(f"✅ Data synchronized in {elapsed:.2f} ms")
            
            return True
        
        return False
    
    def force_sync(self):
        """Принудительная синхронизация"""
        print("\n🔄 Force sync from MongoDB...")
        
        start_time = time.time()
        
        self.analyzer.load_references_from_mongodb(self.db_name)
        self.analyzer.load_diseases_from_mongodb(self.db_name)
        
        metadata = self.db.metadata.find_one({"data_type": "idf_weights"})
        if metadata:
            self.current_version = metadata['version']
        
        elapsed = (time.time() - start_time) * 1000
        print(f"✅ Data synchronized in {elapsed:.2f} ms")


# ============================================================
# Примеры использования
# ============================================================

async def example_async_sync():
    """Пример использования с async/await"""
    from pymongo import MongoClient
    
    # Подключение к MongoDB
    client = MongoClient("mongodb://localhost:27017")
    
    # Создание анализатора с MongoDB клиентом
    analyzer = MedicalLabAnalyzer(mongodb_client=client)
    
    # Создание менеджера синхронизации
    sync_manager = SyncManager(
        analyzer=analyzer,
        mongodb_client=client,
        db_name="medical_lab",
        check_interval=3600  # Проверка каждый час
    )
    
    # Запуск фоновой синхронизации
    sync_task = asyncio.create_task(sync_manager.start())
    
    # Основная работа приложения
    try:
        while True:
            # Тут ваша логика...
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        sync_manager.stop()
        await sync_task


def example_simple_sync():
    """Пример простой синхронизации без async"""
    from pymongo import MongoClient
    
    # Подключение к MongoDB
    client = MongoClient("mongodb://localhost:27017")
    
    # Создание анализатора
    analyzer = MedicalLabAnalyzer(mongodb_client=client)
    
    # Создание простого менеджера
    sync_manager = SimpleSyncManager(
        analyzer=analyzer,
        mongodb_client=client,
        db_name="medical_lab"
    )
    
    # Первоначальная загрузка
    sync_manager.sync_if_changed()
    
    # В основном цикле приложения периодически проверяем
    while True:
        # Ваша логика...
        time.sleep(3600)  # Каждый час
        
        # Проверяем обновления
        if sync_manager.sync_if_changed():
            print("📊 Data was updated!")


def example_fastapi_integration():
    """Пример интеграции с FastAPI"""
    print("""
    # api.py с синхронизацией
    
    from fastapi import FastAPI
    from pymongo import MongoClient
    from disease_search_engine import MedicalLabAnalyzer
    from sync_manager import SyncManager
    
    app = FastAPI()
    
    # Глобальные переменные
    mongodb_client = MongoClient("mongodb://localhost:27017")
    analyzer = MedicalLabAnalyzer(mongodb_client=mongodb_client)
    sync_manager = None
    
    @app.on_event("startup")
    async def startup():
        global sync_manager
        
        # Создаём менеджер синхронизации
        sync_manager = SyncManager(
            analyzer=analyzer,
            mongodb_client=mongodb_client,
            check_interval=3600  # 1 час
        )
        
        # Запускаем фоновую задачу
        asyncio.create_task(sync_manager.start())
    
    @app.on_event("shutdown")
    async def shutdown():
        if sync_manager:
            sync_manager.stop()
    
    @app.post("/api/analyze")
    async def analyze(request: AnalysisRequest):
        results = analyzer.analyze_patient(...)
        return {"results": results}
    
    @app.get("/api/sync/status")
    async def sync_status():
        return sync_manager.get_status()
    
    @app.post("/api/sync/force")
    async def force_sync():
        result = sync_manager.force_sync()
        return result
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("Sync Manager Examples")
    print("=" * 60)
    print("\n1. Async sync (with background task)")
    print("2. Simple sync (periodic check)")
    print("3. FastAPI integration example")
    
    choice = input("\nChoose example to view: ").strip()
    
    if choice == "1":
        print("\n" + "=" * 60)
        print("Running async sync example...")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")
        try:
            asyncio.run(example_async_sync())
        except KeyboardInterrupt:
            print("\nStopped")
    elif choice == "2":
        example_simple_sync()
    elif choice == "3":
        example_fastapi_integration()
