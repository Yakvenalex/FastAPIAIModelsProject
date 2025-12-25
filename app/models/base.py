"""
Базовый класс для всех AI моделей
"""
from abc import ABC, abstractmethod
import torch
import logging

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
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded(),
            "cache_dir": self.cache_dir,
        }
