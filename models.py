"""Интерфейс модели для LegalBench-RU.

Минимальный контракт: у модели есть `.name` и метод `.generate(prompt) -> str`.
Этого достаточно, чтобы прогнать датасет уже сейчас. Полноценные адаптеры
GigaChat / YandexGPT / T-pro в едином eval-интерфейсе — отдельный трек
(libs/ru-evals / DeepEval ru-providers PR); сюда они подключатся как ещё
несколько классов, реализующих тот же протокол.

Без сторонних зависимостей: OpenAI-совместимый адаптер ходит через urllib.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    name: str

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class StubModel:
    """Детерминированная заглушка без сети — для проверки харнесса.

    Ничего не «знает» о праве: возвращает пустую строку. Нужна, чтобы
    убедиться, что пайплайн (load → prompt → score → metrics) работает и что
    скоринг честный (заглушка обязана набрать низкий балл, не случайный высокий).
    """

    name = "stub"

    def generate(self, prompt: str) -> str:
        return ""


class OpenAIModel:
    """Адаптер для любого OpenAI-совместимого chat/completions endpoint.

    Подходит для T-pro, локального vLLM/Ollama, прокси к GigaChat-API и т.п.
    Конфиг через окружение:
        RU_EVAL_BASE_URL  — например https://api.openai.com/v1 или http://localhost:8000/v1
        RU_EVAL_API_KEY   — ключ (если требуется)
    """

    def __init__(self, model_name: str, *, temperature: float = 0.0, timeout: float = 120.0) -> None:
        self.name = model_name
        self._temperature = temperature
        self._timeout = timeout
        self._base_url = os.environ.get("RU_EVAL_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._api_key = os.environ.get("RU_EVAL_API_KEY", "")

    def _call(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"), headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return json.loads(r.read())

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }
        try:
            data = self._call(payload)
        except urllib.error.HTTPError as e:
            # Некоторые модели (напр. реджект temperature=0 у gpt-5-подобных в LiteLLM)
            # → ретрай без temperature.
            if e.code == 400:
                payload.pop("temperature", None)
                data = self._call(payload)
            else:
                raise
        return data["choices"][0]["message"]["content"] or ""


def get_model(kind: str, model_name: str = "") -> Model:
    if kind == "stub":
        return StubModel()
    if kind == "openai":
        if not model_name:
            raise SystemExit("--model openai requires --model-name <name>")
        return OpenAIModel(model_name)
    raise SystemExit(f"unknown model kind: {kind!r} (expected: stub | openai | oracle)")
