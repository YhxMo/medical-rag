"""One OpenAI-compatible client for text generation and screenshot understanding."""

from __future__ import annotations

import json


class ModelClient:
    def __init__(self, config: dict, *, client=None):
        self.config = config
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.config.get("api_key")) or self._client is not None

    def generate(self, system: str, payload, *, images=(), json_output=False) -> str | dict:
        if not self.available:
            raise ValueError("Model API key is missing; configure it or use --offline")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["api_base"],
                timeout=90,
                max_retries=0,
            )
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in images
        ]
        options = {"response_format": {"type": "json_object"}} if json_output else {}
        response = self._client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content if images else text},
            ],
            temperature=0.1,
            max_tokens=self.config.get("max_tokens", 2048),
            **options,
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise RuntimeError("Model returned an incomplete answer")
        if not json_output:
            return choice.message.content
        result = json.loads(choice.message.content)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from the model")
        return result
