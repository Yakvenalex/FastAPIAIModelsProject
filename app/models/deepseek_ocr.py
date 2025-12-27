"""
DeepSeek OCR модель для распознавания текста на изображениях и PDF
"""
from typing import Union, List
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_bytes
import io
import logging
import tempfile
import os
import re
import sys
import gc
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from .base import BaseModel

logger = logging.getLogger(__name__)


@contextmanager
def capture_stdout():
    """Контекстный менеджер для захвата stdout"""
    from io import StringIO
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    try:
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        yield captured_stdout, captured_stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class DeepSeekOCR(BaseModel):
    """
    Модель для распознавания текста (OCR) с использованием DeepSeek-OCR
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: str = "./models_cache",
        base_size: int = 384,
        image_size: int = 256,
        max_image_size: int = 2048,
        use_torch_compile: bool = True,
        use_bfloat16: bool = True
    ):
        super().__init__(model_name, device, cache_dir)
        self.tokenizer = None
        self.base_size = base_size
        self.image_size = image_size
        self.max_image_size = max_image_size
        self.use_torch_compile = use_torch_compile
        self.use_bfloat16 = use_bfloat16
        self._is_compiled = False

    def unload(self):
        """Выгрузка модели из памяти"""
        super().unload()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_compiled = False

    def load(self):
        """Загрузка модели и токенизатора в видеопамять"""
        if self.is_loaded():
            logger.info(f"Модель {self.model_name} уже загружена")
            return

        logger.info(f"Загрузка DeepSeek OCR модели {self.model_name}...")

        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            # Настраиваем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Загружаем модель на GPU с оптимизациями
            torch_dtype = torch.bfloat16 if (self.device == "cuda" and self.use_bfloat16) else torch.float32

            # Настройка max_memory для использования CPU как overflow
            max_memory = None
            if self.device == "cuda":
                from app.config import get_settings
                settings = get_settings()
                gpu_mem = f"{settings.ocr_max_gpu_memory_gb}GB"
                cpu_mem = f"{settings.cpu_offload_memory_gb}GB"
                max_memory = {0: gpu_mem, "cpu": cpu_mem}
                logger.info(f"Настроено распределение памяти OCR: {gpu_mem} GPU + {cpu_mem} CPU overflow")

            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload_cache"  # Папка для offload на диск если нужно
            )

            # Настраиваем generation_config модели
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
                # Убираем несовместимые параметры
                if hasattr(self.model.generation_config, 'temperature'):
                    self.model.generation_config.temperature = None
                if hasattr(self.model.generation_config, 'do_sample'):
                    self.model.generation_config.do_sample = None

            # Переводим модель в режим инференса
            # НЕ вызываем .cuda() - device_map="auto" уже распределил модель
            self.model = self.model.eval()
            logger.info("Модель распределена между GPU и CPU автоматически")

            # Применяем torch.compile для ускорения (если включено)
            if self.use_torch_compile and self.device == "cuda":
                try:
                    logger.info("Применяем torch.compile для ускорения инференса...")
                    # Настраиваем torch._dynamo для подавления ошибок
                    if hasattr(torch, '_dynamo'):
                        torch._dynamo.config.suppress_errors = True
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._is_compiled = True
                    logger.info("torch.compile успешно применён")
                except Exception as e:
                    logger.warning(f"Не удалось применить torch.compile: {e}. Продолжаем без компиляции.")
                    self._is_compiled = False

            logger.info(f"Модель {self.model_name} успешно загружена на {self.device}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def predict(
        self,
        image: Union[Image.Image, bytes],
        is_pdf: bool = False
    ) -> str:
        """
        Распознавание текста на изображении или PDF

        Args:
            image: PIL Image или байты файла
            is_pdf: True если входной файл - PDF

        Returns:
            Распознанный текст
        """
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена. Вызовите load() перед использованием.")

        try:
            # Если это PDF, конвертируем в изображения
            if is_pdf:
                images = self._pdf_to_images(image)
            else:
                # Если переданы байты, конвертируем в PIL Image
                if isinstance(image, bytes):
                    images = [Image.open(io.BytesIO(image)).convert("RGB")]
                else:
                    images = [image.convert("RGB")]

            # Распознаём текст на каждой странице/изображении
            all_text = []
            for idx, img in enumerate(images):
                logger.info(f"Обработка изображения {idx + 1}/{len(images)}")
                text = self._process_single_image(img)
                all_text.append(text)

            # Объединяем текст со всех страниц
            result = "\n\n".join(all_text)
            return result

        except Exception as e:
            logger.error(f"Ошибка при распознавании: {e}")
            raise

    def _pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Конвертация PDF в список изображений"""
        logger.info("Конвертация PDF в изображения...")
        images = convert_from_bytes(pdf_bytes)
        logger.info(f"PDF содержит {len(images)} страниц")
        return images

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """
        Автоматический resize изображения если оно слишком большое
        Сохраняет пропорции изображения
        """
        width, height = image.size
        max_dim = max(width, height)

        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            logger.info(f"Уменьшение изображения с {width}x{height} до {new_width}x{new_height} для ускорения")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _clean_ocr_output(self, text: str) -> str:
        """
        Очистка вывода OCR от служебных тегов и отладочной информации

        Удаляет теги типа <|ref|>, <|det|>, <|grounding|> и координаты,
        а также служебные сообщения модели
        """
        if not text:
            return ""

        # Удаляем строки с отладочной информацией
        lines = text.split('\n')
        filtered_lines = []
        skip_block = False

        for line in lines:
            # Пропускаем блоки между ===
            if '=====' in line:
                skip_block = not skip_block
                continue

            if skip_block:
                continue

            # Пропускаем служебные сообщения
            if any(keyword in line for keyword in [
                'directly resize',
                'BASE:',
                'PATCHES:',
                'NO PATCHES',
                'torch.Size',
                'The attention',
                'seen_tokens',
                'get_max_cache',
                'position_ids',
                'RoPE embeddings'
            ]):
                continue

            filtered_lines.append(line)

        text = '\n'.join(filtered_lines)

        # Удаляем теги с координатами
        text = re.sub(r'<\|det\|>\[\[.*?\]\]<\/\|det\|>', '', text)

        # Удаляем теги ref
        text = re.sub(r'<\|ref\|>.*?<\/\|ref\|>', '', text)

        # Удаляем одиночные теги
        text = re.sub(r'<\|[^>]+\|>', '', text)

        # Удаляем лишние пустые строки
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Убираем пробелы в начале и конце
        text = text.strip()

        return text

    def _process_single_image(self, image: Image.Image) -> str:
        """Обработка одного изображения"""
        import uuid

        # Создаём уникальное имя файла без кириллицы
        unique_id = str(uuid.uuid4())
        tmp_path = os.path.join(tempfile.gettempdir(), f"ocr_{unique_id}.jpg")
        output_dir = os.path.join(tempfile.gettempdir(), f"ocr_out_{unique_id}")

        try:
            # Автоматически уменьшаем изображение если оно слишком большое
            image = self._resize_if_needed(image)

            # Сохраняем изображение
            image.save(tmp_path, format='JPEG')

            # Создаём директорию для результатов
            os.makedirs(output_dir, exist_ok=True)

            # Промпт для простого OCR
            prompt = "<image>\nExtract all text from this image."

            # Логируем использование памяти до инференса
            if self.device == "cuda" and torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"GPU память до инференса: {mem_before:.2f} GB")

            # Инференс с использованием метода модели (захватываем stdout)
            try:
                with capture_stdout() as (captured_out, captured_err):
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        output_path=output_dir,
                        base_size=self.base_size,  # Используем параметр из конфига
                        image_size=self.image_size,  # Используем параметр из конфига
                        crop_mode=False,  # Отключаем crop для скорости
                        save_results=False,
                        test_compress=False
                    )

                # Получаем захваченный вывод
                stdout_content = captured_out.getvalue()

                # Если model.infer вернул None, используем stdout
                if result is None and stdout_content:
                    result = stdout_content
                    logger.info(f"Результат получен из stdout, длина: {len(result)} символов")
                elif result is not None:
                    logger.info(f"Результат получен напрямую, тип: {type(result)}")
                else:
                    logger.warning("model.infer вернул None и stdout пуст")
                    return ""

            except Exception as e:
                logger.error(f"Ошибка при вызове model.infer: {type(e).__name__}: {e}", exc_info=True)
                raise

            # Обрабатываем результат с правильной кодировкой
            try:
                # Убеждаемся, что результат в unicode (str)
                if isinstance(result, bytes):
                    result = result.decode('utf-8', errors='ignore')

                # Принудительно обеспечиваем что это строка
                result = str(result)

            except Exception as e:
                logger.error(f"Ошибка при обработке результата: {type(e).__name__}: {e}")
                raise

            # Очищаем результат от служебных тегов
            cleaned_result = self._clean_ocr_output(result)

            logger.info(f"Распознано {len(cleaned_result)} символов")

            return cleaned_result

        finally:
            # Удаляем временные файлы и директории
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)

            # Агрессивная очистка памяти после каждого инференса
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Логируем использование памяти после очистки
                mem_after = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"GPU память после очистки: {mem_after:.2f} GB")
