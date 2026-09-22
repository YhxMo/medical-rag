"""Shared, process-safe cost reservations and content-addressed model calls."""

from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
import threading
import time
from pathlib import Path

_GLOBAL_MODEL_SLOTS = threading.BoundedSemaphore(2)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class BudgetExceeded(RuntimeError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path, total=30.0, limits=None):
        self.path = Path(path)
        self.total = float(total)
        self.limits = limits or {"ingest": 10.0, "evaluation": 15.0, "demo": 5.0}
        if self.total <= 0 or self.total > 30 or sum(self.limits.values()) > self.total:
            raise ValueError("Budget may not exceed the authorized 30 CNY")
        if any(v <= 0 for v in self.limits.values()):
            raise ValueError("Budget limits must be positive")
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = (
                    json.loads(self.path.read_text())
                    if self.path.exists()
                    else {
                        "total_cny": self.total,
                        "limits": self.limits,
                        "calls": [],
                        "overrun": False,
                    }
                )
                if data["total_cny"] != self.total or data["limits"] != self.limits:
                    raise ValueError(
                        "Budget configuration differs from existing ledger"
                    )
                yield data
                atomic_json(self.path, data)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, category, cost, model):
        if category not in self.limits or cost <= 0:
            raise ValueError("Invalid budget reservation")
        with self.locked() as data:
            used = sum(r["charged_upper_cny"] for r in data["calls"])
            category_used = sum(
                r["charged_upper_cny"]
                for r in data["calls"]
                if r["category"] == category
            )
            if (
                data["overrun"]
                or used + cost > self.total
                or category_used + cost > self.limits[category]
            ):
                raise BudgetExceeded("Authorized API budget exhausted")
            token = len(data["calls"])
            data["calls"].append(
                {
                    "id": token,
                    "category": category,
                    "model": model,
                    "reserved_cny": cost,
                    "charged_upper_cny": cost,
                    "status": "pending",
                    "usage": None,
                }
            )
            return token

    def finish(self, token, *, usage=None, cost=None, status="failed"):
        with self.locked() as data:
            row = data["calls"][token]
            row["status"] = status
            if usage is not None and cost is not None:
                row.update(usage=usage, charged_upper_cny=cost)
                if cost > row["reserved_cny"]:
                    data["overrun"] = (
                        True  # Stop subsequent calls if provider exceeds assumed bound.
                    )


class ModelClient:
    def __init__(self, config, ledger, cache_dir, *, transport=None):
        self.config = config
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.transport = transport
        self._slots = _GLOBAL_MODEL_SLOTS
        self._client = None

    @property
    def available(self):
        return bool(self.config.get("api_key")) or self.transport is not None

    def call(
        self,
        system,
        payload,
        *,
        images=(),
        category="evaluation",
        version="",
        max_tokens=1800,
        json_output=True,
        on_delta=None,
    ):
        if not self.available:
            raise ModelUnavailable(
                "Configure the model API credential in the local environment"
            )
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        image_urls = list(images)
        request_key = digest(
            {
                "endpoint": self.config["api_base"],
                "model": self.config["model"],
                "system": system,
                "payload": text,
                "images": [digest(x) for x in image_urls],
                "version": version,
                "max_tokens": max_tokens,
                "json": json_output,
                "temperature": 0.1,
            }
        )
        cache = self.cache_dir / (request_key + ".json")
        if cache.exists():
            saved = json.loads(cache.read_text())
            if on_delta and isinstance(saved["output"], str):
                on_delta(saved["output"])
            return saved["output"], {
                **saved["metadata"],
                "cache_hit": True,
                "seconds": 0.0,
            }
        if len(image_urls) > 3 or max_tokens > 2048:
            raise ValueError("Request exceeds bounded model context")
        # Upper bound: UTF-8 bytes for text, plus provider maximum 16,384 tokens/image.
        bound = len((system + text).encode()) + 2048 + len(image_urls) * 16384
        if bound > 100000:
            raise ValueError("Input exceeds bounded context")
        input_price = float(self.config["input_cny_per_million"])
        output_price = float(self.config["output_cny_per_million"])
        reserve = (bound * input_price + max_tokens * output_price) / 1e6
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content if image_urls else text},
        ]
        with self._slots:
            for attempt in range(3):
                token = self.ledger.reserve(
                    category if attempt == 0 else "demo", reserve, self.config["model"]
                )
                start = time.perf_counter()
                try:
                    kwargs = dict(
                        model=self.config["model"],
                        messages=messages,
                        temperature=0.1,
                        max_tokens=max_tokens,
                    )
                    if json_output:
                        kwargs["response_format"] = {"type": "json_object"}
                    if "deepseek" in self.config["api_base"]:
                        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                    elif "dashscope" in self.config["api_base"]:
                        kwargs["extra_body"] = {"enable_thinking": False}
                    if on_delta and not json_output:
                        kwargs.update(
                            stream=True, stream_options={"include_usage": True}
                        )
                    if self.transport:
                        response = self.transport(**kwargs)
                    else:
                        if self._client is None:
                            from openai import OpenAI

                            self._client = OpenAI(
                                api_key=self.config["api_key"],
                                base_url=self.config["api_base"],
                                timeout=90,
                                max_retries=0,
                            )
                        response = self._client.chat.completions.create(**kwargs)
                    first_token = None
                    if on_delta and not json_output:
                        from types import SimpleNamespace

                        parts, usage, finish, model = (
                            [],
                            None,
                            None,
                            self.config["model"],
                        )
                        for chunk in response:
                            model = getattr(chunk, "model", None) or model
                            if getattr(chunk, "usage", None) is not None:
                                usage = chunk.usage
                            for choice in chunk.choices:
                                content = getattr(choice.delta, "content", None)
                                if content:
                                    if first_token is None:
                                        first_token = time.perf_counter() - start
                                    parts.append(content)
                                    on_delta(content)
                                if choice.finish_reason:
                                    finish = choice.finish_reason
                        response = SimpleNamespace(
                            model=model,
                            usage=usage,
                            choices=[
                                SimpleNamespace(
                                    finish_reason=finish,
                                    message=SimpleNamespace(content="".join(parts)),
                                )
                            ],
                        )
                    elapsed = time.perf_counter() - start
                    usage = getattr(response, "usage", None)
                    inp, out = (
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                    )
                    measured = (
                        isinstance(inp, int)
                        and isinstance(out, int)
                        and inp >= 0
                        and out >= 0
                    )
                    cost = (
                        (inp * input_price + out * output_price) / 1e6
                        if measured
                        else None
                    )
                    usage_dict = (
                        {"input_tokens": inp, "output_tokens": out}
                        if measured
                        else None
                    )
                    self.ledger.finish(
                        token, usage=usage_dict, cost=cost, status="response_received"
                    )
                    choice = response.choices[0]
                    if choice.finish_reason != "stop" or not choice.message.content:
                        atomic_json(
                            self.cache_dir
                            / "failed-responses"
                            / f"{request_key}-{token}.json",
                            {
                                "finish_reason": choice.finish_reason,
                                "content": choice.message.content,
                                "usage": usage_dict,
                                "max_tokens": max_tokens,
                                "model": response.model,
                            },
                        )
                    if choice.finish_reason != "stop" or not choice.message.content:
                        raise ModelUnavailable(
                            "Incomplete model output: " + str(choice.finish_reason)
                        )
                    output = (
                        json.loads(choice.message.content)
                        if json_output
                        else choice.message.content
                    )
                    if json_output and not isinstance(output, dict):
                        raise ValueError("Model JSON must be an object")
                    metadata = {
                        "seconds": elapsed,
                        "usage": usage_dict,
                        "cost_upper_cny": cost,
                        "model": response.model,
                        "cache_hit": False,
                        "first_token_seconds": first_token,
                        "streaming": bool(on_delta),
                        "output_token_limit": max_tokens,
                    }
                    atomic_json(cache, {"output": output, "metadata": metadata})
                    return output, metadata
                except Exception as exc:
                    self.ledger.finish(token, status="failed")
                    if isinstance(exc, ModelUnavailable):
                        raise
                    status = getattr(exc, "status_code", None)
                    transient = status in {429, 500, 502, 503, 504} or type(
                        exc
                    ).__name__ in {
                        "APITimeoutError",
                        "APIConnectionError",
                        "TimeoutError",
                    }
                    if not transient or attempt == 2 or on_delta:
                        raise ModelUnavailable(
                            f"Model request failed ({type(exc).__name__})"
                        ) from None
                    time.sleep(2**attempt)
        raise ModelUnavailable("Model request failed")
