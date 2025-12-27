"""
Роутер для OCR (распознавание текста)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Dict
import logging
import asyncio

from app.models.deepseek_ocr import DeepSeekOCR
from app.config import get_settings
from app.utils import log_gpu_memory, auto_unload_old_models_if_needed

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
            cache_dir=settings.cache_dir,
            base_size=settings.ocr_base_size,
            image_size=settings.ocr_image_size,
            max_image_size=settings.ocr_max_image_size,
            use_torch_compile=settings.ocr_use_torch_compile,
            use_bfloat16=settings.ocr_use_bfloat16
        )
        # Загружаем только если не используется lazy loading
        if not settings.lazy_loading:
            ocr_model.load()
    return ocr_model


@router.post("/extract-text")
async def extract_text(
    file: UploadFile = File(..., description="Изображение (PNG, JPG, WebP) или PDF файл"),
    normalize: bool = False
) -> Dict:
    """
    Распознавание текста на изображении или PDF файле

    Args:
        file: Загруженный файл (изображение PNG, JPG, WebP или PDF)
        normalize: Если True, текст будет нормализован через LLM (исправление ошибок, форматирование)

    Returns:
        Словарь с результатами:
        {
            "text": "Оригинальный распознанный текст",
            "normalized_text": "Нормализованный текст" или null,
            "normalized": true/false,
            "status": "success",
            "content_type": "image/jpeg",
            "filename": "file.jpg"
        }
    """
    # Проверяем тип файла
    allowed_types = {
        "image/png", "image/jpeg", "image/jpg", "image/webp",
        "application/pdf"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат файла. Разрешены: PNG, JPG, WebP, PDF"
        )

    try:
        # Читаем файл
        contents = await file.read()
        logger.info(f"Получен файл: {file.filename}, размер: {len(contents)} байт")

        # Проверяем размер файла (50MB лимит)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Файл слишком большой. Максимальный размер: 50MB"
            )

        # Определяем тип файла
        is_pdf = file.content_type == "application/pdf"

        # Получаем модель и выполняем распознавание
        model = get_ocr_model()

        # Загружаем модель если еще не загружена (lazy loading)
        if not model.is_loaded():
            logger.info("Загрузка OCR модели по требованию...")

            # КРИТИЧНО: Проверяем память и выгружаем старые модели если нужно
            await asyncio.to_thread(auto_unload_old_models_if_needed, required_gb=2.5)

            await asyncio.to_thread(model.load)
            log_gpu_memory("После загрузки OCR:")

        # Обновляем время последнего использования
        model.update_last_used()

        # Распознавание текста в отдельном потоке (не блокирует event loop)
        original_text = await asyncio.to_thread(model.predict, contents, is_pdf)

        normalized_text = None
        # Нормализация текста через Chat модель если запрошено
        if normalize and original_text and original_text.strip():
            from app.routers.chat import get_chat_model

            logger.info("Нормализация распознанного текста через Chat модель...")
            chat_model = get_chat_model()

            # Загружаем chat модель если нужно
            if not chat_model.is_loaded():
                logger.info("Загрузка Chat модели для нормализации...")
                await asyncio.to_thread(auto_unload_old_models_if_needed, required_gb=3.0)
                await asyncio.to_thread(chat_model.load)
                log_gpu_memory("После загрузки Chat для нормализации:")

            chat_model.update_last_used()

            # Промпт для нормализации OCR текста
            normalization_prompt = """Ты опытный редактор текста. Твоя задача - исправить и улучшить текст, распознанный с помощью OCR.

Что нужно сделать:
1. Исправить явные ошибки распознавания (перепутанные буквы, цифры вместо букв и т.д.)
2. Восстановить правильную пунктуацию и заглавные буквы
3. Убрать лишние пробелы и переносы строк
4. Сохранить структуру текста (абзацы, списки)
5. НЕ добавлять новую информацию, только исправлять ошибки

Верни ТОЛЬКО исправленный текст, без комментариев и пояснений."""

            # Нормализация в отдельном потоке (не блокирует event loop)
            normalized_text = await asyncio.to_thread(
                chat_model.chat,
                user_message=f"Текст для нормализации:\n\n{original_text}",
                system_prompt=normalization_prompt,
                temperature=0.3,  # Низкая температура для точности
                max_new_tokens=2048  # Достаточно для длинных текстов
            )
            normalized_text = normalized_text.strip()
            logger.info("Текст успешно нормализован")

        return {
            "text": original_text,
            "normalized_text": normalized_text,
            "normalized": normalize,
            "status": "success",
            "content_type": file.content_type,
            "filename": file.filename
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
