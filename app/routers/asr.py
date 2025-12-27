"""
Роутер для ASR (автоматическое распознавание речи)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Dict
import logging
import asyncio

from app.models.whisper_asr import WhisperASR
from app.config import get_settings
from app.utils import log_gpu_memory, auto_unload_old_models_if_needed

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
        # Загружаем только если не используется lazy loading
        if not settings.lazy_loading:
            asr_model.load()
    return asr_model


@router.post("/transcribe", response_model=Dict)
async def transcribe_audio(
    file: UploadFile = File(..., description="Аудиофайл (WAV, MP3, FLAC, OGG, M4A, и др.)"),
    normalize: bool = False
) -> Dict:
    """
    Распознавание речи из аудиофайла (автоматическое распознавание речи)

    Поддерживает аудиофайлы любой длины. Длинные аудио автоматически разбиваются
    на чанки для обработки.

    Args:
        file: Загруженный аудиофайл (поддерживаются форматы: WAV, MP3, FLAC, OGG, M4A и др.)
        normalize: Если True, текст будет нормализован через LLM (пунктуация, форматирование)

    Returns:
        Словарь с результатами распознавания:
        {
            "text": "Оригинальный распознанный текст",
            "normalized_text": "Нормализованный текст" или null,
            "normalized": true/false,
            "status": "success",
            "content_type": "audio/mpeg",
            "filename": "audio.mp3",
            "duration": 123.45,
            "num_chunks": 3,
            "chunks": [
                {
                    "chunk_index": 0,
                    "start_time": 0.0,
                    "end_time": 60.0,
                    "text": "Оригинальный текст чанка" (только если normalize=false),
                    "original_text": "Оригинальный текст чанка" (если normalize=true),
                    "normalized_text": "Нормализованный текст чанка" (если normalize=true)
                },
                ...
            ]
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

        # Загружаем модель если еще не загружена (lazy loading)
        if not model.is_loaded():
            logger.info("Загрузка ASR модели по требованию...")

            # КРИТИЧНО: Проверяем память и выгружаем старые модели если нужно
            await asyncio.to_thread(auto_unload_old_models_if_needed, required_gb=2.5)

            await asyncio.to_thread(model.load)
            log_gpu_memory("После загрузки ASR:")

        # Обновляем время последнего использования
        model.update_last_used()

        # Распознавание речи в отдельном потоке (не блокирует event loop)
        result = await asyncio.to_thread(model.predict, contents, file.filename or "")

        # Сохраняем оригинальные тексты
        original_full_text = result.get("full_text", "")
        normalized_full_text = None

        # Нормализация текста через Chat модель если запрошено
        if normalize and original_full_text and original_full_text.strip():
            from app.routers.chat import get_chat_model

            logger.info("Нормализация распознанной речи через Chat модель...")
            chat_model = get_chat_model()

            # Загружаем chat модель если нужно
            if not chat_model.is_loaded():
                logger.info("Загрузка Chat модели для нормализации...")
                await asyncio.to_thread(auto_unload_old_models_if_needed, required_gb=3.0)
                await asyncio.to_thread(chat_model.load)
                log_gpu_memory("После загрузки Chat для нормализации:")

            chat_model.update_last_used()

            # Промпт для нормализации ASR текста
            normalization_prompt = """Ты опытный редактор текста. Твоя задача - исправить и улучшить текст, полученный из системы распознавания речи.

Что нужно сделать:
1. Добавить правильную пунктуацию (точки, запятые, знаки вопроса и т.д.)
2. Расставить заглавные буквы в начале предложений и в именах собственных
3. Исправить очевидные ошибки распознавания слов
4. Разбить текст на абзацы если это уместно
5. Убрать слова-паразиты и повторы ТОЛЬКО если они явно мешают пониманию
6. НЕ менять смысл и содержание, только улучшать читаемость

Верни ТОЛЬКО исправленный текст, без комментариев и пояснений."""

            # Нормализация полного текста в отдельном потоке
            normalized_full_text = await asyncio.to_thread(
                chat_model.chat,
                user_message=f"Текст для нормализации:\n\n{original_full_text}",
                system_prompt=normalization_prompt,
                temperature=0.3,  # Низкая температура для точности
                max_new_tokens=2048
            )
            normalized_full_text = normalized_full_text.strip()

            # Также нормализуем текст в чанках если они есть
            if "chunks" in result and result["chunks"]:
                for chunk in result["chunks"]:
                    if chunk.get("text"):
                        chunk["original_text"] = chunk["text"]
                        chunk_normalized = await asyncio.to_thread(
                            chat_model.chat,
                            user_message=f"Текст для нормализации:\n\n{chunk['text']}",
                            system_prompt=normalization_prompt,
                            temperature=0.3,
                            max_new_tokens=1024
                        )
                        chunk["normalized_text"] = chunk_normalized.strip()

            logger.info("Текст успешно нормализован")

        # Формируем финальный результат
        result["text"] = original_full_text
        result["normalized_text"] = normalized_full_text
        result["normalized"] = normalize
        result["status"] = "success"
        result["content_type"] = file.content_type
        result["filename"] = file.filename

        # Удаляем старое поле full_text для консистентности
        if "full_text" in result:
            del result["full_text"]

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
