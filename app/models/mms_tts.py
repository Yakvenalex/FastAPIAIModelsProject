"""
MMS TTS модель для синтеза русской речи
"""
from typing import Dict, Tuple
import torch
from transformers import AutoTokenizer, VitsModel, AutoProcessor
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
from pathlib import Path

from .base import BaseModel

logger = logging.getLogger(__name__)


class MMSTTS(BaseModel):
    """
    Модель для синтеза речи (TTS) с использованием Facebook MMS
    Оптимизирована для русского языка
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: str = "./models_cache",
        use_bfloat16: bool = True,
        sample_rate: int = 16000
    ):
        """
        Инициализация модели MMS TTS

        Args:
            model_name: Название модели на HuggingFace
            device: Устройство для запуска ("cuda" или "cpu")
            cache_dir: Директория для кэша моделей
            use_bfloat16: Использовать bfloat16 для экономии памяти
            sample_rate: Частота дискретизации аудио (обычно 16000 для MMS)
        """
        super().__init__(model_name, device, cache_dir)
        self.use_bfloat16 = use_bfloat16
        self.sample_rate = sample_rate
        self.tokenizer = None

    def load(self):
        """Загрузка модели и токенизатора в видеопамять"""
        if self.is_loaded():
            logger.info(f"Модель {self.model_name} уже загружена")
            return

        logger.info(f"Загрузка MMS TTS модели {self.model_name}...")

        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # VITS модель НЕ поддерживает device_map="auto"
            # Загружаем напрямую на устройство
            self.model = VitsModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True
            )

            # Переносим на устройство вручную
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                logger.info(f"TTS модель загружена на {self.device}")

            # Переводим модель в режим инференса
            self.model.eval()
            logger.info("MMS TTS модель распределена между GPU и CPU")

            logger.info(f"Модель {self.model_name} успешно загружена на {self.device}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def unload(self):
        """Выгрузка модели из памяти"""
        super().unload()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

    def predict(self, text: str, output_dir: str = None) -> Tuple[str, Dict]:
        """
        Синтез речи из текста

        Args:
            text: Текст для синтеза речи
            output_dir: Директория для сохранения WAV файла (по умолчанию temp)

        Returns:
            Tuple[str, Dict]: Путь к WAV файлу и метаданные
            {
                "file_path": "/path/to/audio.wav",
                "duration": 3.45,
                "sample_rate": 16000,
                "num_samples": 55200,
                "text": "исходный текст"
            }
        """
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена. Вызовите load() перед использованием.")

        if not text or not text.strip():
            raise ValueError("Текст не может быть пустым")

        try:
            logger.info(f"Синтез речи для текста длиной {len(text)} символов")

            # Токенизация текста
            inputs = self.tokenizer(text, return_tensors="pt")

            # Переносим на устройство (но НЕ меняем dtype - токены должны быть int64)
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Генерируем аудио
            with torch.no_grad():
                output = self.model(**inputs)

            # Получаем waveform
            # output.waveform имеет форму [batch_size, sequence_length]
            waveform = output.waveform[0].cpu().numpy()

            # Нормализуем аудио к диапазону int16
            # VITS обычно выдает аудио в диапазоне [-1, 1]
            waveform = np.clip(waveform, -1.0, 1.0)
            audio_int16 = (waveform * 32767).astype(np.int16)

            # Определяем директорию для сохранения
            if output_dir is None:
                output_dir = tempfile.gettempdir()

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Генерируем уникальное имя файла
            import uuid
            file_id = str(uuid.uuid4())
            file_path = output_dir / f"tts_{file_id}.wav"

            # Сохраняем WAV файл
            wavfile.write(str(file_path), self.sample_rate, audio_int16)

            duration = len(audio_int16) / self.sample_rate

            logger.info(f"Аудио сохранено: {file_path}, длительность: {duration:.2f}s")

            # Очистка памяти
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            metadata = {
                "file_path": str(file_path),
                "duration": round(duration, 2),
                "sample_rate": self.sample_rate,
                "num_samples": len(audio_int16),
                "text": text
            }

            return str(file_path), metadata

        except Exception as e:
            logger.error(f"Ошибка при синтезе речи: {e}")
            raise

    def get_info(self) -> dict:
        """Получить информацию о модели"""
        info = super().get_info()
        info.update({
            "sample_rate": self.sample_rate,
            "use_bfloat16": self.use_bfloat16,
            "language": "ru",
            "task": "text-to-speech"
        })
        return info
