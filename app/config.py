"""
Конфигурация приложения
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Настройки приложения"""

    # Основные настройки
    app_name: str = "FastAPI AI Models"
    app_version: str = "1.0.0"
    debug: bool = False

    # Настройки для моделей
    device: str = "cuda"  # "cuda" или "cpu"
    cache_dir: str = "./models_cache"  # директория для кэша моделей

    # Настройки для DeepSeek OCR
    deepseek_ocr_model: str = "deepseek-ai/DeepSeek-OCR"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ()  # отключаем защиту namespace
    }


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки (синглтон)"""
    return Settings()
