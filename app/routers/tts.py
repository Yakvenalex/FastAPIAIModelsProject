"""
Роутер для TTS (синтез речи)
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict
import logging
import os
from pathlib import Path

from app.models.mms_tts import MMSTTS
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["TTS (Text-to-Speech)"])

# Глобальный экземпляр модели (загружается при старте приложения)
tts_model: MMSTTS = None


def get_tts_model() -> MMSTTS:
    """Получить экземпляр TTS модели"""
    global tts_model
    if tts_model is None:
        settings = get_settings()
        tts_model = MMSTTS(
            model_name=settings.mms_tts_model,
            device=settings.device,
            cache_dir=settings.cache_dir,
            use_bfloat16=settings.tts_use_bfloat16,
            sample_rate=settings.tts_sample_rate
        )
        tts_model.load()
    return tts_model


class TTSRequest(BaseModel):
    """Запрос на синтез речи"""
    text: str = Field(..., description="Текст для синтеза речи", min_length=1)


def cleanup_file(file_path: str):
    """Фоновая задача для удаления файла после отправки"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Файл удалён: {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при удалении файла {file_path}: {e}")


@router.post("/synthesize")
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Синтез речи из текста

    Возвращает WAV аудиофайл для прямого скачивания.
    Файл автоматически удаляется после отправки.

    Args:
        request: Объект с текстом для синтеза
        background_tasks: FastAPI background tasks для удаления файла

    Returns:
        WAV аудиофайл
    """
    try:
        # Получаем модель и синтезируем речь
        model = get_tts_model()
        file_path, metadata = model.predict(request.text)

        # Добавляем задачу на удаление файла после отправки
        background_tasks.add_task(cleanup_file, file_path)

        # Возвращаем файл для скачивания
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename="speech.wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"'
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ошибка при синтезе речи: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при синтезе речи: {str(e)}"
        )


@router.get("/model-info", response_model=Dict)
async def get_model_info() -> Dict:
    """
    Получить информацию о загруженной TTS модели

    Returns:
        Словарь с информацией о модели
    """
    model = get_tts_model()
    return model.get_info()


@router.post("/reload-model")
async def reload_model() -> Dict[str, str]:
    """
    Перезагрузить TTS модель

    Returns:
        Статус операции
    """
    global tts_model
    try:
        if tts_model is not None:
            tts_model.unload()
            tts_model = None

        # Загружаем модель заново
        get_tts_model()

        return {"status": "success", "message": "Модель успешно перезагружена"}

    except Exception as e:
        logger.error(f"Ошибка при перезагрузке модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при перезагрузке модели: {str(e)}"
        )
