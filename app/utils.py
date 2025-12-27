"""
Утилиты для управления памятью и мониторинга
"""
import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Получить информацию об использовании GPU памяти

    Returns:
        Dict с информацией о памяти в GB
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "total_gb": 0.0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "free_gb": 0.0
        }

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    free = total - allocated

    return {
        "available": True,
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "usage_percent": round((allocated / total) * 100, 1)
    }


def log_gpu_memory(prefix: str = ""):
    """
    Логировать текущее использование GPU памяти

    Args:
        prefix: Префикс для лога
    """
    info = get_gpu_memory_info()
    if info["available"]:
        msg = (
            f"{prefix} GPU память: "
            f"{info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
            f"({info['usage_percent']:.1f}%), "
            f"свободно: {info['free_gb']:.2f}GB"
        )
        logger.info(msg)
    else:
        logger.info(f"{prefix} GPU недоступен")


def clear_gpu_cache():
    """
    Очистить кэш GPU памяти
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU кэш очищен")


def optimize_gpu_memory():
    """
    Агрессивная оптимизация GPU памяти
    """
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Дополнительно: сброс фрагментации памяти
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        logger.debug("GPU память оптимизирована")


def check_gpu_memory_available(required_gb: float = 2.0) -> bool:
    """
    Проверить, достаточно ли свободной GPU памяти

    Args:
        required_gb: Требуемое количество свободной памяти в GB

    Returns:
        True если достаточно памяти
    """
    if not torch.cuda.is_available():
        return True  # На CPU нет ограничений

    info = get_gpu_memory_info()
    free_gb = info.get("free_gb", 0)

    return free_gb >= required_gb


def ensure_gpu_memory(required_gb: float = 2.0, force_cleanup: bool = True) -> bool:
    """
    Убедиться что достаточно GPU памяти, при необходимости очистить

    Args:
        required_gb: Требуемое количество свободной памяти в GB
        force_cleanup: Принудительно очистить память если не хватает

    Returns:
        True если удалось обеспечить достаточно памяти
    """
    if not torch.cuda.is_available():
        return True

    # Проверяем текущее состояние
    if check_gpu_memory_available(required_gb):
        return True

    if not force_cleanup:
        return False

    # Пытаемся освободить память
    logger.info(f"Недостаточно GPU памяти (требуется {required_gb}GB), выполняем очистку...")
    optimize_gpu_memory()

    # Проверяем снова после очистки
    return check_gpu_memory_available(required_gb)


def auto_unload_old_models_if_needed(required_gb: float = 2.0):
    """
    Автоматически выгрузить старые модели если не хватает памяти

    Args:
        required_gb: Требуемое количество свободной памяти в GB
    """
    if not torch.cuda.is_available():
        return

    # Проверяем, нужна ли выгрузка
    if check_gpu_memory_available(required_gb):
        return

    logger.warning(f"Недостаточно GPU памяти! Требуется {required_gb}GB, начинаем выгрузку старых моделей...")

    # Импортируем модели
    try:
        from app.routers import ocr, asr, tts, chat
    except ImportError:
        logger.error("Не удалось импортировать роутеры для выгрузки моделей")
        return

    # Собираем все загруженные модели с временем последнего использования
    models_info = []

    if ocr.ocr_model and ocr.ocr_model.is_loaded():
        models_info.append(("OCR", ocr.ocr_model, ocr.ocr_model.get_idle_time_minutes()))
    if asr.asr_model and asr.asr_model.is_loaded():
        models_info.append(("ASR", asr.asr_model, asr.asr_model.get_idle_time_minutes()))
    if tts.tts_model and tts.tts_model.is_loaded():
        models_info.append(("TTS", tts.tts_model, tts.tts_model.get_idle_time_minutes()))
    if chat.chat_model and chat.chat_model.is_loaded():
        models_info.append(("Chat", chat.chat_model, chat.chat_model.get_idle_time_minutes()))

    if not models_info:
        logger.warning("Нет загруженных моделей для выгрузки")
        return

    # Сортируем по времени простоя (от большего к меньшему)
    models_info.sort(key=lambda x: x[2], reverse=True)

    # Выгружаем модели по одной, пока не освободим достаточно памяти
    for name, model, idle_time in models_info:
        logger.info(f"Выгрузка {name} модели (простой: {idle_time:.1f} минут)")
        model.unload()
        optimize_gpu_memory()

        # Проверяем, достаточно ли памяти теперь
        if check_gpu_memory_available(required_gb):
            logger.info(f"Достаточно памяти после выгрузки {name}")
            return

    logger.warning("Выгружены все модели, но памяти всё ещё недостаточно")
