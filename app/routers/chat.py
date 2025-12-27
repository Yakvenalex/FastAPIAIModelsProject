"""
Роутер для Chat (LLM чат-бот)
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Optional
import logging
import asyncio

from app.models.qwen_chat import LLMChat
from app.config import get_settings
from app.utils import log_gpu_memory, auto_unload_old_models_if_needed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat (LLM)"])

# Глобальный экземпляр модели (загружается при старте приложения)
chat_model: LLMChat = None


def get_chat_model() -> LLMChat:
    """Получить экземпляр Chat модели"""
    global chat_model
    if chat_model is None:
        settings = get_settings()
        chat_model = LLMChat(
            model_name=settings.gemma_chat_model,
            device=settings.device,
            cache_dir=settings.cache_dir,
            use_bfloat16=settings.chat_use_bfloat16,
            use_4bit=settings.chat_use_4bit,
            max_new_tokens=settings.chat_max_new_tokens,
            temperature=settings.chat_temperature,
            top_p=settings.chat_top_p,
            max_memory_gb=settings.chat_max_gpu_memory_gb
        )
        # Загружаем только если не используется lazy loading
        if not settings.lazy_loading:
            chat_model.load()
    return chat_model


class ChatRequest(BaseModel):
    """Запрос для чата"""
    message: str = Field(..., description="Сообщение пользователя", min_length=1)
    system_prompt: Optional[str] = Field(
        default="Ты полезный AI-ассистент. Всегда отвечай на русском языке, будь вежливым и помогай пользователю.",
        description="Системный промпт (инструкция для модели)"
    )
    max_new_tokens: Optional[int] = Field(default=None, description="Максимальное количество токенов в ответе")
    temperature: Optional[float] = Field(default=None, ge=0.1, le=1.0, description="Температура генерации (0.1-1.0)")


class ChatResponse(BaseModel):
    """Ответ от чат-модели"""
    response: str
    status: str


@router.post("/simple", response_model=ChatResponse)
async def simple_chat(request: ChatRequest) -> ChatResponse:
    """
    Простой чат (одно сообщение)

    Используйте этот endpoint для простых вопросов или задач:
    - "Улучши этот текст: ..."
    - "Переведи на английский: ..."
    - "Объясни что такое ..."
    - "Напиши код для ..."

    Args:
        request: Запрос с сообщением пользователя

    Returns:
        Ответ модели
    """
    try:
        model = get_chat_model()

        # Загружаем модель если еще не загружена (lazy loading)
        if not model.is_loaded():
            logger.info("Загрузка Chat модели по требованию...")

            # КРИТИЧНО: Проверяем память и выгружаем старые модели если нужно
            await asyncio.to_thread(auto_unload_old_models_if_needed, required_gb=3.0)

            await asyncio.to_thread(model.load)
            log_gpu_memory("После загрузки Chat:")

        # Обновляем время последнего использования
        model.update_last_used()

        # Chat генерация в отдельном потоке (не блокирует event loop)
        response = await asyncio.to_thread(
            model.chat,
            user_message=request.message,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        return ChatResponse(
            response=response,
            status="success"
        )

    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации ответа: {str(e)}"
        )


@router.get("/model-info", response_model=Dict)
async def get_model_info() -> Dict:
    """
    Получить информацию о загруженной Chat модели

    Returns:
        Словарь с информацией о модели
    """
    model = get_chat_model()
    return model.get_info()


@router.post("/reload-model")
async def reload_model() -> Dict[str, str]:
    """
    Перезагрузить Chat модель

    Returns:
        Статус операции
    """
    global chat_model
    try:
        if chat_model is not None:
            chat_model.unload()
            chat_model = None

        # Загружаем модель заново
        get_chat_model()

        return {"status": "success", "message": "Модель успешно перезагружена"}

    except Exception as e:
        logger.error(f"Ошибка при перезагрузке модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при перезагрузке модели: {str(e)}"
        )
