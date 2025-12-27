"""
Конфигурация приложения
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

# Оптимизация PyTorch для уменьшения фрагментации памяти
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    ocr_base_size: int = 384  # Уменьшен для экономии памяти (было 512)
    ocr_image_size: int = 256  # Уменьшен для экономии памяти (было 384)
    ocr_max_image_size: int = 2048  # Максимальный размер изображения (уменьшен с 3072)
    ocr_use_torch_compile: bool = False  # Отключено - нестабильно с device_map="auto"
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
    gemma_chat_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Qwen2.5-3B - компактная но качественная модель
    chat_use_bfloat16: bool = True  # Использовать bfloat16 для экономии памяти
    chat_use_4bit: bool = False  # БЕЗ квантизации - лучшее качество для 3B модели!
    chat_max_new_tokens: int = 512  # Максимальное количество генерируемых токенов
    chat_temperature: float = 0.5  # Температура генерации (креативность), диапазон 0.1-1.0
    chat_top_p: float = 0.9  # Top-p sampling параметр

    # Настройки управления памятью
    lazy_loading: bool = True  # Загружать модели по требованию
    auto_unload: bool = True  # Автоматически выгружать неиспользуемые модели
    auto_unload_timeout_minutes: int = 5  # Выгружать модель после 5 минут простоя

    # Динамическое распределение памяти между моделями
    # OCR и Whisper - самые тяжелые, TTS - легкая, Chat - средняя
    ocr_max_gpu_memory_gb: float = 10.0  # DeepSeek OCR (большая vision модель)
    whisper_max_gpu_memory_gb: float = 10.0  # Whisper Large-v3 (большая)
    tts_max_gpu_memory_gb: float = 8.0  # MMS TTS (маленькая, много места не нужно)
    chat_max_gpu_memory_gb: float = 12.0  # Qwen2.5-3B (средняя)

    cpu_offload_memory_gb: float = 25.0  # Использовать до 25GB из 32GB RAM для overflow
    enable_cpu_offload: bool = True  # Использовать CPU RAM для overflow

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
