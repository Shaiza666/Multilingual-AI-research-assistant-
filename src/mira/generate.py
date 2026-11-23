"""Prompt construction and pluggable generation.

The default backend is a local GGUF instruct model via llama.cpp. A
transformers model or an OpenAI-compatible HTTP server can be selected with
``MIRA_LLM_BACKEND``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from pathlib import Path

from mira.config import Config
from mira.errors import MiraError
from mira.grounding import ABSTAIN_TOKEN

logger = logging.getLogger("mira.generate")

PROMPT_VERSION = "mira-prompt-v3"

_SYSTEM = (
    "You answer questions about research papers using only the passages. "
    f"If the passages do not contain the answer, reply with exactly {ABSTAIN_TOKEN}. "
    "Answer in the same language as the question, in one short sentence. "
    "Keep person names, dataset names, model names, numbers, and technical terms "
    "in Latin script exactly as they appear in the passages. "
    "Do not repeat or translate the question. "
    "After the answer, cite one passage id in square brackets. "
    "Do not use knowledge that is not in the passages."
)


def build_messages(question: str, language: str, passages: list[tuple[str, str]]) -> list[dict]:
    lang_name = "Urdu" if language == "ur" else "English"
    if language == "ur":
        example_user = (
            "Passages:\n"
            "[demo::p1::c0] ALFA stands for Aligned Latent Factor Averaging and uses a learning rate of 0.005.\n\n"
            "Question (Urdu): ALFA کس کا مخفف ہے اور لرننگ ریٹ کیا ہے؟\n\n"
            "Write one short Urdu sentence. Keep names and numbers in Latin script. "
            f"Cite one passage id. If the answer is missing, reply {ABSTAIN_TOKEN}."
        )
        example_assistant = "ALFA کا مطلب Aligned Latent Factor Averaging ہے اور لرننگ ریٹ 0.005 ہے۔ [demo::p1::c0]"
    else:
        example_user = (
            "Passages:\n"
            "[demo::p1::c0] Batch normalization normalizes layer inputs using the mean and variance of the current mini-batch.\n\n"
            "Question (English): What statistics does batch normalization use?\n\n"
            f"Write the fact in English. Do not repeat the question. Cite one passage id. "
            f"If the answer is not in the passages, reply {ABSTAIN_TOKEN}."
        )
        example_assistant = (
            "Batch normalization uses the mean and variance of the current mini-batch. [demo::p1::c0]"
        )
    blocks = [f"[{chunk_id}] {text}" for chunk_id, text in passages]
    user = (
        "Passages:\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion ({lang_name}): {question}\n\n"
        + f"Write one short sentence in {lang_name}, at most 20 words. "
        + "Keep names and numbers in Latin script. Do not repeat the question. Cite one passage id. "
        + f"If the answer is not in the passages, reply {ABSTAIN_TOKEN}."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": user},
    ]


def fit_passages(passages: list[tuple[str, str]], max_chars: int, budget_chars: int) -> list[tuple[str, str]]:
    clipped = [(chunk_id, text[:max_chars]) for chunk_id, text in passages]
    total = sum(len(text) for _cid, text in clipped)
    if total <= budget_chars or not clipped:
        return clipped
    per = max(180, budget_chars // len(clipped))
    return [(chunk_id, text[:per]) for chunk_id, text in clipped]


def prompt_cache_key(cfg: Config, messages: list[dict]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "backend": cfg.llm.backend,
        "model": _model_identity(cfg),
        "max_new_tokens": cfg.llm.max_new_tokens,
        "temperature": cfg.llm.temperature,
        "messages": messages,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_identity(cfg: Config) -> str:
    if cfg.llm.backend == "openai":
        return cfg.llm.openai_model
    if cfg.llm.backend == "transformers":
        return cfg.llm.transformers_model
    return str(cfg.llm.gguf_path)


class GenerationCache:
    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, str] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._data[row["key"]] = row["text"]

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, text: str) -> None:
        self._data[key] = text
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


def generate(cfg: Config, messages: list[dict], cache: GenerationCache | None = None) -> tuple[str, bool]:
    key = prompt_cache_key(cfg, messages)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached, True
    backend = cfg.llm.backend
    if backend == "local-gguf":
        text = _generate_gguf(cfg, messages)
    elif backend == "transformers":
        text = _generate_transformers(cfg, messages)
    elif backend == "openai":
        text = _generate_openai(cfg, messages)
    else:
        raise MiraError(f"Unknown LLM backend: {backend}")
    text = text.strip()
    if cache is not None:
        cache.put(key, text)
    return text, False


def _llama(cfg: Config):
    from mira.models import _CACHE

    path = Path(cfg.llm.gguf_path)
    if not path.exists():
        raise MiraError(
            f"Local GGUF model not found at {path}. Run `python scripts/download_models.py` "
            "or set MIRA_LLM_BACKEND=transformers."
        )
    key = ("llm", str(path.resolve()))
    if key not in _CACHE:
        from llama_cpp import Llama

        logger.info("Loading GGUF model %s", path)
        _CACHE[key] = Llama(
            model_path=str(path),
            n_ctx=cfg.llm.n_ctx,
            n_threads=cfg.llm.n_threads,
            n_batch=128,
            verbose=False,
        )
    return _CACHE[key]


def _generate_gguf(cfg: Config, messages: list[dict]) -> str:
    llm = _llama(cfg)
    kwargs = {
        "messages": messages,
        "temperature": cfg.llm.temperature,
        "max_tokens": cfg.llm.max_new_tokens,
        "repeat_penalty": cfg.llm.repeat_penalty,
        "seed": cfg.llm.seed,
    }
    try:
        result = llm.create_chat_completion(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        result = llm.create_chat_completion(**kwargs)
    return result["choices"][0]["message"]["content"]


def _generate_transformers(cfg: Config, messages: list[dict]) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mira.models import _CACHE, _torch_threads

    key = ("llm", cfg.llm.transformers_model)
    if key not in _CACHE:
        _torch_threads(cfg.llm.n_threads)
        logger.info("Loading transformers model %s on CPU", cfg.llm.transformers_model)
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm.transformers_model)
        model = AutoModelForCausalLM.from_pretrained(cfg.llm.transformers_model)
        model.eval()
        _CACHE[key] = (tokenizer, model)
    tokenizer, model = _CACHE[key]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.llm.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generate_openai(cfg: Config, messages: list[dict]) -> str:
    base = cfg.llm.openai_base_url.rstrip("/")
    if not base or not cfg.llm.openai_api_key or not cfg.llm.openai_model:
        raise MiraError("OpenAI-compatible generation needs OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL.")
    url = base + "/chat/completions"
    body = json.dumps(
        {
            "model": cfg.llm.openai_model,
            "messages": messages,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_new_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.llm.openai_api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]
