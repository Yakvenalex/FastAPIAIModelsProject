"""
Главный файл FastAPI приложения с AI моделями
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.routers import ocr

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Получаем настройки
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup: загружаем все модели в память при старте
    logger.info(f"Запуск {settings.app_name} v{settings.app_version}")
    logger.info(f"Используемое устройство: {settings.device}")
    logger.info(f"Директория кэша моделей: {settings.cache_dir}")

    logger.info("Загрузка моделей в память...")
    # Предзагружаем OCR модель
    ocr.get_ocr_model()
    logger.info("Все модели загружены и готовы к работе!")

    yield

    # Shutdown: выгружаем модели из памяти
    logger.info("Остановка приложения...")
    if ocr.ocr_model is not None:
        ocr.ocr_model.unload()
    logger.info("Приложение остановлено")


# Создаём FastAPI приложение
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI приложение для работы с AI моделями (DeepSeek OCR и другие)",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(ocr.router)


@app.get("/", tags=["Health"])
async def root():
    """Корневой endpoint"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "available_endpoints": {
            "ocr": "/ocr/extract-text - распознавание текста на изображениях и PDF",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья приложения"""
    return {
        "status": "healthy",
        "device": settings.device,
    }
