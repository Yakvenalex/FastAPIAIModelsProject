"""
Роутер для Chat (LLM чат-бот)
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import logging

from app.models.gemma_chat import LLMChat
from app.config import get_settings

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
            max_new_tokens=settings.chat_max_new_tokens,
            temperature=settings.chat_temperature,
            top_p=settings.chat_top_p
        )
        chat_model.load()
    return chat_model


class SimpleChatRequest(BaseModel):
    """Простой запрос для чата (одно сообщение)"""
    message: str = Field(..., description="Сообщение пользователя", min_length=1)
    system_prompt: Optional[str] = Field(
        default="Ты полезный AI-ассистент. Всегда отвечай на русском языке, будь вежливым и помогай пользователю.",
        description="Системный промпт (инструкция для модели)"
    )
    max_new_tokens: Optional[int] = Field(default=None, description="Максимальное количество токенов в ответе")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Температура генерации (0-2)")


class ChatMessage(BaseModel):
    """Сообщение в диалоге"""
    role: str = Field(..., description="Роль отправителя (user, assistant, system)")
    content: str = Field(..., description="Содержимое сообщения")


class MultiTurnChatRequest(BaseModel):
    """Запрос для мультиоборотного чата (с историей)"""
    messages: List[ChatMessage] = Field(..., description="История диалога", min_length=1)
    max_new_tokens: Optional[int] = Field(default=None, description="Максимальное количество токенов в ответе")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Температура генерации (0-2)")


class ChatResponse(BaseModel):
    """Ответ от чат-модели"""
    response: str
    status: str


@router.post("/simple", response_model=ChatResponse)
async def simple_chat(request: SimpleChatRequest) -> ChatResponse:
    """
    Простой чат (одно сообщение без истории)

    Используйте этот endpoint для простых вопросов или задач типа:
    - "Улучши этот текст: ..."
    - "Переведи на английский: ..."
    - "Объясни что такое ..."

    Args:
        request: Запрос с сообщением пользователя

    Returns:
        Ответ модели
    """
    try:
        model = get_chat_model()

        response = model.chat(
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


@router.post("/multi-turn", response_model=ChatResponse)
async def multi_turn_chat(request: MultiTurnChatRequest) -> ChatResponse:
    """
    Мультиоборотный чат (с историей диалога)

    Используйте для продолжения диалога с сохранением контекста.

    Пример истории:
    [
        {"role": "system", "content": "Ты помощник программиста"},
        {"role": "user", "content": "Как создать список в Python?"},
        {"role": "assistant", "content": "Список создаётся так: my_list = []"},
        {"role": "user", "content": "А как добавить элемент?"}
    ]

    Args:
        request: Запрос с историей диалога

    Returns:
        Ответ модели
    """
    try:
        model = get_chat_model()

        # Преобразуем Pydantic модели в словари для процессора
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        response = model.predict(
            messages=messages,
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
