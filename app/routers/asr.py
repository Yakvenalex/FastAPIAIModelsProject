"""
Роутер для ASR (автоматическое распознавание речи)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Dict
import logging

from app.models.whisper_asr import WhisperASR
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/asr", tags=["ASR (Speech Recognition)"])

# Глобальный экземпляр модели (загружается при старте приложения)
asr_model: WhisperASR = None


def get_asr_model() -> WhisperASR:
    """Получить экземпляр ASR модели"""
    global asr_model
    if asr_model is None:
        settings = get_settings()
        asr_model = WhisperASR(
            model_name=settings.whisper_model,
            device=settings.device,
            cache_dir=settings.cache_dir,
            use_bfloat16=settings.whisper_use_bfloat16,
            chunk_length_s=settings.whisper_chunk_length_s,
            batch_size=settings.whisper_batch_size,
            use_bettertransformer=settings.whisper_use_bettertransformer
        )
        asr_model.load()
    return asr_model


@router.post("/transcribe", response_model=Dict)
async def transcribe_audio(
    file: UploadFile = File(..., description="Аудиофайл (WAV, MP3, FLAC, OGG, M4A, и др.)")
) -> Dict:
    """
    Распознавание речи из аудиофайла (автоматическое распознавание речи)

    Поддерживает аудиофайлы любой длины. Длинные аудио автоматически разбиваются
    на чанки для обработки.

    Args:
        file: Загруженный аудиофайл (поддерживаются форматы: WAV, MP3, FLAC, OGG, M4A и др.)

    Returns:
        Словарь с результатами распознавания:
        {
            "filename": "audio.mp3",
            "content_type": "audio/mpeg",
            "full_text": "Полный распознанный текст",
            "duration": 123.45,
            "num_chunks": 3,
            "chunks": [
                {
                    "chunk_index": 0,
                    "start_time": 0.0,
                    "end_time": 60.0,
                    "text": "Текст первой минуты"
                },
                ...
            ],
            "status": "success"
        }
    """
    # Проверяем тип файла (поддерживаем основные аудио форматы)
    allowed_types = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/x-flac",
        "audio/ogg", "audio/x-ogg",
        "audio/m4a", "audio/x-m4a",
        "audio/aac",
        "audio/webm",
        "audio/opus",
        "application/octet-stream",  # Некоторые браузеры отправляют так
    }

    if file.content_type and file.content_type not in allowed_types:
        # Проверяем по расширению файла как fallback
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".opus"}
        file_ext = None
        if file.filename:
            import os
            file_ext = os.path.splitext(file.filename)[1].lower()

        if not file_ext or file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый формат файла. Разрешены: WAV, MP3, FLAC, OGG, M4A, AAC, WebM, Opus"
            )

    try:
        # Читаем файл
        contents = await file.read()
        logger.info(f"Получен аудиофайл: {file.filename}, размер: {len(contents)} байт")

        # Проверяем размер файла (100MB лимит для аудио)
        max_size = 100 * 1024 * 1024  # 100MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Файл слишком большой. Максимальный размер: 100MB"
            )

        # Получаем модель и выполняем распознавание
        model = get_asr_model()
        result = model.predict(contents, filename=file.filename or "")

        # Добавляем метаинформацию
        result["filename"] = file.filename
        result["content_type"] = file.content_type
        result["status"] = "success"

        return result

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )


@router.get("/model-info", response_model=Dict)
async def get_model_info() -> Dict:
    """
    Получить информацию о загруженной ASR модели

    Returns:
        Словарь с информацией о модели
    """
    model = get_asr_model()
    return model.get_info()


@router.post("/reload-model")
async def reload_model() -> Dict[str, str]:
    """
    Перезагрузить ASR модель

    Returns:
        Статус операции
    """
    global asr_model
    try:
        if asr_model is not None:
            asr_model.unload()
            asr_model = None

        # Загружаем модель заново
        get_asr_model()

        return {"status": "success", "message": "Модель успешно перезагружена"}

    except Exception as e:
        logger.error(f"Ошибка при перезагрузке модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при перезагрузке модели: {str(e)}"
        )
