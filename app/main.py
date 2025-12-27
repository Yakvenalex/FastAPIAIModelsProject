"""
Главный файл FastAPI приложения с AI моделями
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.routers import ocr, asr, tts, chat
from app.memory_manager import memory_manager
from app.utils import log_gpu_memory, get_gpu_memory_info

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
    # Startup: загружаем модели в память при старте
    logger.info(f"Запуск {settings.app_name} v{settings.app_version}")
    logger.info(f"Используемое устройство: {settings.device}")
    logger.info(f"Директория кэша моделей: {settings.cache_dir}")
    logger.info(f"Lazy loading: {'включен' if settings.lazy_loading else 'выключен'}")
    logger.info(f"Auto-unload: {'включен' if settings.auto_unload else 'выключен'}")

    # Логируем начальное состояние GPU памяти
    log_gpu_memory("Начальное состояние:")

    if settings.lazy_loading:
        logger.info("Модели будут загружены по требованию (lazy loading)")
        # Инициализируем модели но не загружаем их
        ocr.get_ocr_model()
        asr.get_asr_model()
        tts.get_tts_model()
        chat.get_chat_model()
        logger.info("Модели инициализированы, готовы к загрузке по требованию")
    else:
        logger.info("Загрузка всех моделей в память...")
        # Предзагружаем все модели
        ocr_model = ocr.get_ocr_model()
        ocr_model.load()
        log_gpu_memory("После загрузки OCR:")

        asr_model = asr.get_asr_model()
        asr_model.load()
        log_gpu_memory("После загрузки ASR:")

        tts_model = tts.get_tts_model()
        tts_model.load()
        log_gpu_memory("После загрузки TTS:")

        chat_model = chat.get_chat_model()
        chat_model.load()
        log_gpu_memory("После загрузки Chat:")

        logger.info("Все модели загружены и готовы к работе!")

    # Запускаем менеджер памяти для автоматической выгрузки
    if settings.auto_unload:
        await memory_manager.start()

    yield

    # Shutdown: останавливаем менеджер и выгружаем модели
    logger.info("Остановка приложения...")

    if settings.auto_unload:
        await memory_manager.stop()

    if ocr.ocr_model is not None:
        ocr.ocr_model.unload()
    if asr.asr_model is not None:
        asr.asr_model.unload()
    if tts.tts_model is not None:
        tts.tts_model.unload()
    if chat.chat_model is not None:
        chat.chat_model.unload()

    log_gpu_memory("После выгрузки всех моделей:")
    logger.info("Приложение остановлено")


# Создаём FastAPI приложение
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI приложение для работы с AI моделями (DeepSeek OCR, Whisper ASR, MMS TTS, Qwen2.5 Chat)",
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
            "chat": "/chat/simple - чат с AI ассистентом (Qwen2.5-3B)",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья приложения"""
    gpu_info = get_gpu_memory_info()

    return {
        "status": "healthy",
        "device": settings.device,
        "gpu_memory": gpu_info if gpu_info["available"] else "N/A",
        "models_loaded": {
            "ocr": ocr.ocr_model.is_loaded() if ocr.ocr_model else False,
            "asr": asr.asr_model.is_loaded() if asr.asr_model else False,
            "tts": tts.tts_model.is_loaded() if tts.tts_model else False,
            "chat": chat.chat_model.is_loaded() if chat.chat_model else False,
        }
    }


@app.get("/models/status", tags=["Health"])
async def models_status():
    """Получить подробный статус всех моделей"""
    models_info = {}

    if ocr.ocr_model:
        models_info["ocr"] = ocr.ocr_model.get_info()
    if asr.asr_model:
        models_info["asr"] = asr.asr_model.get_info()
    if tts.tts_model:
        models_info["tts"] = tts.tts_model.get_info()
    if chat.chat_model:
        models_info["chat"] = chat.chat_model.get_info()

    return {
        "models": models_info,
        "gpu_memory": get_gpu_memory_info(),
        "settings": {
            "lazy_loading": settings.lazy_loading,
            "auto_unload": settings.auto_unload,
            "auto_unload_timeout_minutes": settings.auto_unload_timeout_minutes if settings.auto_unload else None
        }
    }
