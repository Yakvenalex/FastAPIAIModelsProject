"""
Whisper ASR модель для распознавания русской речи
"""
from typing import List, Dict, Tuple
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import logging
import tempfile
import os
import librosa
import soundfile as sf
import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)


class WhisperASR(BaseModel):
    """
    Модель для автоматического распознавания речи (ASR) с использованием Whisper
    Оптимизирована для русского языка
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: str = "./models_cache",
        use_bfloat16: bool = True,
        chunk_length_s: int = 60,
        batch_size: int = 8,
        use_bettertransformer: bool = True
    ):
        """
        Инициализация модели Whisper

        Args:
            model_name: Название модели на HuggingFace
            device: Устройство для запуска ("cuda" или "cpu")
            cache_dir: Директория для кэша моделей
            use_bfloat16: Использовать bfloat16 для экономии памяти
            chunk_length_s: Длина чанка в секундах для длинных аудио
            batch_size: Размер батча для обработки чанков
            use_bettertransformer: Использовать BetterTransformer для ускорения
        """
        super().__init__(model_name, device, cache_dir)
        self.use_bfloat16 = use_bfloat16
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.sample_rate = 16000  # Whisper работает с 16kHz
        self.use_bettertransformer = use_bettertransformer

    def load(self):
        """Загрузка модели и процессора в видеопамять"""
        if self.is_loaded():
            logger.info(f"Модель {self.model_name} уже загружена")
            return

        logger.info(f"Загрузка Whisper ASR модели {self.model_name}...")

        try:
            # Загружаем процессор
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # Определяем тип данных
            torch_dtype = torch.bfloat16 if (self.device == "cuda" and self.use_bfloat16) else torch.float32

            # Настройка max_memory для CPU overflow
            max_memory = None
            if self.device == "cuda":
                from app.config import get_settings
                settings = get_settings()
                gpu_mem = f"{settings.whisper_max_gpu_memory_gb}GB"
                cpu_mem = f"{settings.cpu_offload_memory_gb}GB"
                max_memory = {0: gpu_mem, "cpu": cpu_mem}
                logger.info(f"Настроено распределение памяти Whisper: {gpu_mem} GPU + {cpu_mem} CPU overflow")

            # Загружаем модель на GPU с автоматическим распределением
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map="auto",
                max_memory=max_memory
            )

            # Переводим модель в режим инференса
            self.model.eval()
            logger.info("Whisper модель распределена между GPU и CPU")

            # Применяем BetterTransformer для ускорения (до 1.5x быстрее)
            if self.use_bettertransformer:
                try:
                    self.model = self.model.to_bettertransformer()
                    logger.info("Применён BetterTransformer для ускорения инференса")
                except Exception as e:
                    logger.warning(f"Не удалось применить BetterTransformer: {e}")

            # Настраиваем generation config
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.language = "ru"
                self.model.generation_config.task = "transcribe"
                # Оптимизация: используем greedy decoding для скорости
                self.model.generation_config.num_beams = 1  # Greedy search быстрее beam search
                self.model.generation_config.do_sample = False

            logger.info(f"Модель {self.model_name} успешно загружена на {self.device}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def _load_audio(self, audio_data: bytes, original_filename: str = "") -> Tuple[np.ndarray, int]:
        """
        Загрузка аудио из байтов с автоматической конвертацией в нужный формат

        Args:
            audio_data: Байты аудиофайла
            original_filename: Оригинальное имя файла для определения формата

        Returns:
            Tuple[np.ndarray, int]: Аудио массив и частота дискретизации
        """
        import uuid

        # Создаём временный файл
        suffix = os.path.splitext(original_filename)[1] if original_filename else ".tmp"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=tempfile.gettempdir()
        )

        try:
            # Записываем байты во временный файл
            temp_file.write(audio_data)
            temp_file.flush()
            temp_file.close()

            # Загружаем аудио с помощью librosa (поддерживает все форматы)
            # Автоматически ресэмплируем до 16kHz (требование Whisper)
            audio, sr = librosa.load(temp_file.name, sr=self.sample_rate, mono=True)

            logger.info(f"Загружено аудио: {len(audio) / sr:.2f} секунд, sample_rate={sr}")

            return audio, sr

        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def _split_audio_into_chunks(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Разделение длинного аудио на чанки

        Args:
            audio: Аудио массив
            sr: Частота дискретизации

        Returns:
            Список чанков аудио
        """
        duration = len(audio) / sr
        chunk_samples = self.chunk_length_s * sr

        if duration <= self.chunk_length_s:
            # Аудио короткое, возвращаем как есть
            return [audio]

        # Разбиваем на чанки
        chunks = []
        num_chunks = int(np.ceil(duration / self.chunk_length_s))

        logger.info(f"Разделение аудио длительностью {duration:.2f}с на {num_chunks} чанков по {self.chunk_length_s}с")

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[start:end]
            chunks.append(chunk)

        return chunks

    def _transcribe_chunk(self, audio_chunk: np.ndarray, sr: int) -> str:
        """
        Распознавание одного чанка аудио

        Args:
            audio_chunk: Чанк аудио
            sr: Частота дискретизации

        Returns:
            Распознанный текст
        """
        # Подготавливаем входные данные
        inputs = self.processor(
            audio_chunk,
            sampling_rate=sr,
            return_tensors="pt"
        )

        # Переносим на устройство
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if self.use_bfloat16:
                # Преобразуем float тензоры в bfloat16
                inputs = {
                    k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }

        # Генерируем транскрипцию
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)

        # Декодируем результат
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()

    def predict(self, audio_data: bytes, filename: str = "") -> Dict[str, any]:
        """
        Распознавание речи из аудиофайла

        Args:
            audio_data: Байты аудиофайла (поддерживается wav, mp3, flac, ogg, m4a и др.)
            filename: Оригинальное имя файла

        Returns:
            Словарь с результатами:
            {
                "full_text": "полный текст",
                "duration": 123.45,
                "chunks": [
                    {
                        "start_time": 0,
                        "end_time": 60,
                        "text": "текст первой минуты"
                    },
                    ...
                ]
            }
        """
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена. Вызовите load() перед использованием.")

        try:
            # Загружаем аудио
            audio, sr = self._load_audio(audio_data, filename)
            duration = len(audio) / sr

            # Разбиваем на чанки
            chunks = self._split_audio_into_chunks(audio, sr)

            # Распознаём каждый чанк
            results = []
            full_texts = []

            for idx, chunk in enumerate(chunks):
                start_time = idx * self.chunk_length_s
                end_time = min((idx + 1) * self.chunk_length_s, duration)

                logger.info(f"Распознавание чанка {idx + 1}/{len(chunks)} ({start_time:.1f}s - {end_time:.1f}s)")

                text = self._transcribe_chunk(chunk, sr)
                full_texts.append(text)

                results.append({
                    "chunk_index": idx,
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "text": text
                })

            # Объединяем весь текст
            full_text = " ".join(full_texts)

            # Очистка памяти
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "full_text": full_text,
                "duration": round(duration, 2),
                "num_chunks": len(chunks),
                "chunks": results
            }

        except Exception as e:
            logger.error(f"Ошибка при распознавании: {e}")
            raise

    def get_info(self) -> dict:
        """Получить информацию о модели"""
        info = super().get_info()
        info.update({
            "chunk_length_s": self.chunk_length_s,
            "sample_rate": self.sample_rate,
            "use_bfloat16": self.use_bfloat16,
            "language": "ru",
            "task": "transcribe"
        })
        return info
