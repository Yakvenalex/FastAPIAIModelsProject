"""
Роутер для OCR (распознавание текста)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Dict
import logging

from app.models.deepseek_ocr import DeepSeekOCR
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr", tags=["OCR"])

# Глобальный экземпляр модели (загружается при старте приложения)
ocr_model: DeepSeekOCR = None


def get_ocr_model() -> DeepSeekOCR:
    """Получить экземпляр OCR модели"""
    global ocr_model
    if ocr_model is None:
        settings = get_settings()
        ocr_model = DeepSeekOCR(
            model_name=settings.deepseek_ocr_model,
            device=settings.device,
            cache_dir=settings.cache_dir
        )
        ocr_model.load()
    return ocr_model


@router.post("/extract-text", response_model=Dict[str, str])
async def extract_text(
    file: UploadFile = File(..., description="Изображение (PNG, JPG) или PDF файл")
) -> Dict[str, str]:
    """
    Распознавание текста на изображении или PDF файле

    Args:
        file: Загруженный файл (изображение или PDF)

    Returns:
        Словарь с распознанным текстом и информацией о файле
    """
    # Проверяем тип файла
    allowed_types = {
        "image/png", "image/jpeg", "image/jpg",
        "application/pdf"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат файла. Разрешены: PNG, JPG, PDF"
        )

    try:
        # Читаем файл
        contents = await file.read()
        logger.info(f"Получен файл: {file.filename}, размер: {len(contents)} байт")

        # Определяем тип файла
        is_pdf = file.content_type == "application/pdf"

        # Получаем модель и выполняем распознавание
        model = get_ocr_model()
        text = model.predict(contents, is_pdf=is_pdf)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "text": text,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )


@router.get("/model-info", response_model=Dict)
async def get_model_info() -> Dict:
    """
    Получить информацию о загруженной OCR модели

    Returns:
        Словарь с информацией о модели
    """
    model = get_ocr_model()
    return model.get_info()


@router.post("/reload-model")
async def reload_model() -> Dict[str, str]:
    """
    Перезагрузить OCR модель

    Returns:
        Статус операции
    """
    global ocr_model
    try:
        if ocr_model is not None:
            ocr_model.unload()
            ocr_model = None

        # Загружаем модель заново
        get_ocr_model()

        return {"status": "success", "message": "Модель успешно перезагружена"}

    except Exception as e:
        logger.error(f"Ошибка при перезагрузке модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при перезагрузке модели: {str(e)}"
        )
