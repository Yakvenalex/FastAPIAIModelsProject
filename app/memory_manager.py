"""
Менеджер памяти для автоматического управления моделями
"""
import asyncio
import logging
from typing import List
from app.config import get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Менеджер для автоматической выгрузки неиспользуемых моделей
    """

    def __init__(self):
        self.settings = get_settings()
        self._running = False
        self._task = None

    async def start(self):
        """Запустить фоновую задачу мониторинга"""
        if self._running:
            logger.warning("MemoryManager уже запущен")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"MemoryManager запущен (таймаут: {self.settings.auto_unload_timeout_minutes} минут)")

    async def stop(self):
        """Остановить фоновую задачу"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MemoryManager остановлен")

    async def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self._running:
            try:
                # Проверяем каждую минуту
                await asyncio.sleep(60)

                if not self.settings.auto_unload:
                    continue

                # Импортируем модели здесь чтобы избежать циклических импортов
                from app.routers import ocr, asr, tts, chat

                models_to_check = [
                    ("OCR", ocr.ocr_model),
                    ("ASR", asr.asr_model),
                    ("TTS", tts.tts_model),
                    ("Chat", chat.chat_model)
                ]

                for name, model in models_to_check:
                    if model is not None and model.should_auto_unload(self.settings.auto_unload_timeout_minutes):
                        idle_time = model.get_idle_time_minutes()
                        logger.info(f"Автоматическая выгрузка {name} модели (простой: {idle_time:.1f} минут)")
                        model.unload()

                        # Логируем освободившуюся память
                        from app.utils import log_gpu_memory
                        log_gpu_memory(f"После выгрузки {name}:")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в MemoryManager: {e}", exc_info=True)


# Глобальный экземпляр менеджера
memory_manager = MemoryManager()
