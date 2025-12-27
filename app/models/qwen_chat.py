"""
Универсальная LLM модель для чат-задач
"""
from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class LLMChat(BaseModel):
    """
    Универсальная модель для чат-задач (Mistral, Gemma, и другие CausalLM)
    Поддерживает текстовые запросы с chat template
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: str = "./models_cache",
        use_bfloat16: bool = True,
        use_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_memory_gb: float = None
    ):
        """
        Инициализация LLM модели

        Args:
            model_name: Название модели на HuggingFace
            device: Устройство для запуска ("cuda" или "cpu")
            cache_dir: Директория для кэша моделей
            use_bfloat16: Использовать bfloat16 для экономии памяти
            use_4bit: Использовать 4-bit квантизацию (сильная экономия памяти)
            max_new_tokens: Максимальное количество генерируемых токенов
            temperature: Температура генерации (креативность)
            top_p: Top-p sampling параметр
            max_memory_gb: Максимальная память GPU в GB (для ограничения)
        """
        super().__init__(model_name, device, cache_dir)
        self.use_bfloat16 = use_bfloat16
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_memory_gb = max_memory_gb
        self.tokenizer = None

    def load(self):
        """Загрузка модели и токенизатора"""
        if self.is_loaded():
            logger.info(f"Модель {self.model_name} уже загружена")
            return

        logger.info(f"Загрузка LLM модели {self.model_name}...")
        if self.use_4bit:
            logger.info("Используется 4-bit квантизация для экономии памяти")

        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # Конфигурация квантизации
            quantization_config = None
            if self.use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bfloat16 else torch.float16,
                    bnb_4bit_use_double_quant=True,  # Двойная квантизация для еще большей экономии
                    bnb_4bit_quant_type="nf4"  # NormalFloat4 - лучший тип для LLM
                )

            # Определяем тип данных (если не используется квантизация)
            torch_dtype = None
            if not self.use_4bit:
                torch_dtype = torch.bfloat16 if (self.device == "cuda" and self.use_bfloat16) else torch.float32

            # Настраиваем max_memory для контроля памяти
            max_memory = None
            if self.device == "cuda":
                from app.config import get_settings
                settings = get_settings()
                # Используем либо переданный параметр, либо из конфига
                gpu_mem_gb = self.max_memory_gb if self.max_memory_gb else settings.chat_max_gpu_memory_gb
                cpu_mem_gb = settings.cpu_offload_memory_gb
                max_memory = {0: f"{gpu_mem_gb}GB", "cpu": f"{cpu_mem_gb}GB"}
                logger.info(f"Настроено распределение памяти Chat: {gpu_mem_gb}GB GPU + {cpu_mem_gb}GB CPU overflow")

            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else "cpu",
                max_memory=max_memory
            )

            # Переводим модель в режим инференса
            self.model.eval()

            logger.info(f"Модель {self.model_name} успешно загружена на {self.device}")
            if self.use_4bit:
                logger.info("Модель квантизована в 4-bit (экономия ~75% памяти)")
            elif self.use_bfloat16 and self.device == "cuda":
                logger.info("Модель использует bfloat16 для экономии памяти")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def unload(self):
        """Выгрузка модели из памяти"""
        super().unload()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

    def predict(
        self,
        messages: List[Dict[str, Union[str, List[Dict]]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Генерация ответа на основе диалоговых сообщений

        Args:
            messages: Список сообщений в формате чата
                [
                    {
                        "role": "user",
                        "content": "текст" или [{"type": "text", "text": "..."}, {"type": "image", "image": PIL.Image}]
                    }
                ]
            max_new_tokens: Максимальное количество генерируемых токенов
            temperature: Температура генерации
            top_p: Top-p sampling параметр

        Returns:
            Сгенерированный текст ответа
        """
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена. Вызовите load() перед использованием.")

        try:
            # Используем параметры по умолчанию если не указаны
            max_new_tokens = max_new_tokens or self.max_new_tokens
            temperature = temperature or self.temperature
            top_p = top_p or self.top_p

            logger.info(f"Генерация ответа для {len(messages)} сообщений")

            # Применяем chat template и токенизируем
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Переносим на устройство
            if self.device == "cuda":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Генерируем ответ с оптимизированными параметрами
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Оптимизации для предотвращения зависания
                    num_beams=1,  # Greedy decoding быстрее
                    early_stopping=True,  # Остановка при достижении EOS
                    use_cache=True,  # Использовать KV-cache для ускорения
                    repetition_penalty=1.1  # Предотвращение зацикливания
                )

            # Декодируем только новые токены (без входного промпта)
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logger.info(f"Сгенерирован ответ длиной {len(response)} символов")

            # Очистка памяти
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response.strip()

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Упрощённый метод для одиночного сообщения (без истории)

        Args:
            user_message: Сообщение пользователя
            system_prompt: Системный промпт (опционально)
            max_new_tokens: Максимальное количество генерируемых токенов
            temperature: Температура генерации

        Returns:
            Ответ модели
        """
        messages = []

        # Добавляем системный промпт если есть
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Добавляем сообщение пользователя
        messages.append({
            "role": "user",
            "content": user_message
        })

        return self.predict(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    def get_info(self) -> dict:
        """Получить информацию о модели"""
        info = super().get_info()
        info.update({
            "use_bfloat16": self.use_bfloat16,
            "use_4bit": self.use_4bit,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_memory_gb": self.max_memory_gb,
            "task": "chat"
        })
        return info
