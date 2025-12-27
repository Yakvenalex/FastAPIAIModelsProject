"""
Базовый класс для всех AI моделей
"""
from abc import ABC, abstractmethod
import torch
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Базовый класс для всех моделей.
    Обеспечивает единый интерфейс для загрузки и использования моделей.
    """

    def __init__(self, model_name: str, device: str = "cuda", cache_dir: str = "./models_cache"):
        """
        Инициализация модели

        Args:
            model_name: Название модели на HuggingFace
            device: Устройство для запуска ("cuda" или "cpu")
            cache_dir: Директория для кэша моделей
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir

        if self.device == "cpu" and device == "cuda":
            logger.warning("CUDA недоступна, используется CPU")

        self.model = None
        self.processor = None
        self.last_used: Optional[datetime] = None  # Время последнего использования

        logger.info(f"Инициализация модели {model_name} на устройстве {self.device}")

    @abstractmethod
    def load(self):
        """
        Загрузка модели в память и на устройство.
        Должен быть реализован в наследниках.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Выполнение инференса.
        Должен быть реализован в наследниках.
        """
        pass

    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель"""
        return self.model is not None

    def update_last_used(self):
        """Обновить время последнего использования"""
        self.last_used = datetime.now()

    def get_idle_time_minutes(self) -> float:
        """Получить время простоя в минутах"""
        if self.last_used is None:
            return 0.0
        delta = datetime.now() - self.last_used
        return delta.total_seconds() / 60.0

    def should_auto_unload(self, timeout_minutes: int) -> bool:
        """
        Проверить, нужно ли автоматически выгрузить модель

        Args:
            timeout_minutes: Таймаут в минутах

        Returns:
            True если модель загружена и не использовалась больше timeout_minutes
        """
        if not self.is_loaded():
            return False
        if self.last_used is None:
            return False
        return self.get_idle_time_minutes() > timeout_minutes

    def unload(self):
        """Выгрузка модели из памяти"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Модель {self.model_name} выгружена из памяти")

    def get_info(self) -> dict:
        """Получить информацию о модели"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded(),
            "cache_dir": self.cache_dir,
        }
        if self.last_used:
            info["last_used"] = self.last_used.isoformat()
            info["idle_time_minutes"] = round(self.get_idle_time_minutes(), 2)
        return info
