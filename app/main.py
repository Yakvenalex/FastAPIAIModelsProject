"""
Главный файл FastAPI приложения с AI моделями
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.routers import ocr, asr, tts, chat

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
    # Предзагружаем ASR модель
    asr.get_asr_model()
    # Предзагружаем TTS модель
    tts.get_tts_model()
    # Предзагружаем Chat модель
    chat.get_chat_model()
    logger.info("Все модели загружены и готовы к работе!")

    yield

    # Shutdown: выгружаем модели из памяти
    logger.info("Остановка приложения...")
    if ocr.ocr_model is not None:
        ocr.ocr_model.unload()
    if asr.asr_model is not None:
        asr.asr_model.unload()
    if tts.tts_model is not None:
        tts.tts_model.unload()
    if chat.chat_model is not None:
        chat.chat_model.unload()
    logger.info("Приложение остановлено")


# Создаём FastAPI приложение
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI приложение для работы с AI моделями (DeepSeek OCR, Whisper ASR и другие)",
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
app.include_router(asr.router)
app.include_router(tts.router)
app.include_router(chat.router)


@app.get("/", tags=["Health"])
async def root():
    """Корневой endpoint"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "available_endpoints": {
            "ocr": "/ocr/extract-text - распознавание текста на изображениях (PNG, JPG, WebP) и PDF",
            "asr": "/asr/transcribe - распознавание речи из аудиофайлов (WAV, MP3, FLAC, OGG, M4A и др.)",
            "tts": "/tts/synthesize - синтез речи из текста (Text-to-Speech)",
            "chat": "/chat/simple - чат с AI ассистентом (вопросы, улучшение текста и т.д.)",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья приложения"""
    return {
        "status": "healthy",
        "device": settings.device,
    }
