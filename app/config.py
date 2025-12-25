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

    # Параметры производительности OCR
    ocr_base_size: int = 512  # Баланс между скоростью и качеством
    ocr_image_size: int = 384  # Баланс между скоростью и качеством
    ocr_max_image_size: int = 3072  # Максимальный размер изображения для автоматического resize
    ocr_use_torch_compile: bool = False  # torch.compile может замедлить первый запуск
    ocr_use_bfloat16: bool = True  # Использовать bfloat16 для экономии памяти и ускорения

    # Настройки для Whisper ASR
    whisper_model: str = "antony66/whisper-large-v3-russian"
    whisper_use_bfloat16: bool = True  # Использовать bfloat16 для экономии памяти
    whisper_chunk_length_s: int = 60  # Длина чанка для длинных аудио (в секундах)
    whisper_batch_size: int = 8  # Размер батча для обработки
    whisper_use_bettertransformer: bool = True  # Использовать BetterTransformer для ускорения

    # Настройки для MMS TTS
    mms_tts_model: str = "facebook/mms-tts-rus"
    tts_use_bfloat16: bool = False  # VITS модель не поддерживает bfloat16 стабильно
    tts_sample_rate: int = 16000  # Частота дискретизации аудио (Hz)

    # Настройки для Chat LLM
    gemma_chat_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    chat_use_bfloat16: bool = True  # Использовать bfloat16 для экономии памяти
    chat_max_new_tokens: int = 512  # Максимальное количество генерируемых токенов
    chat_temperature: float = 0.7  # Температура генерации (креативность)
    chat_top_p: float = 0.9  # Top-p sampling параметр

    # Ограничения для загрузки файлов
    max_file_size_mb: int = 50  # Максимальный размер файла в MB

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ()  # отключаем защиту namespace
    }


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки (синглтон)"""
    return Settings()
