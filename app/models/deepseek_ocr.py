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

    def __init__(self, model_name: str, device: str = "cuda", cache_dir: str = "./models_cache"):
        super().__init__(model_name, device, cache_dir)
        self.tokenizer = None

    def unload(self):
        """Выгрузка модели из памяти"""
        super().unload()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

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

            # Загружаем модель на GPU
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
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

            # Переводим модель в режим инференса и на устройство
            self.model = self.model.eval()
            if self.device == "cuda":
                self.model = self.model.cuda()

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
                        base_size=512,  # Уменьшено с 1024 для скорости
                        image_size=384,  # Уменьшено с 640 для скорости
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
