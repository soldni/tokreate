import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

# import aiohttp
import openai

from .base import BaseProvider, ProviderMessage, ProviderRegistry, ProviderResult


@dataclass(frozen=True, eq=True)
class OpenAIResult(ProviderResult):
    function_call: dict = field(default_factory=dict)


class OpenAI(BaseProvider):
    model: Optional[str] = None  # type: ignore
    client_args: Optional[dict] = None

    def __init__(self, api_key: Optional[str] = None, client_args: Optional[dict] = None) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        (client_args := dict(**(client_args or {}))).update(self.client_args or {})
        self.client = openai.OpenAI(api_key=api_key, **client_args)

    def _prepare_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        messages = [ProviderMessage(role="user", content=prompt)]

        if history:
            messages = [*[ProviderMessage(**m) for m in history], ProviderMessage(role="user", content=prompt)]

        if system_message:
            messages = [*messages, ProviderMessage(role="system", content=system_message)]

        model_inputs = {
            "messages": [m.asdict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        return model_inputs

    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
        **kwargs,
    ) -> OpenAIResult:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency() as latency:
            response: Any = self.client.chat.completions.create(model=self.model, **model_inputs)  # type: ignore

        is_func_call = response.choices[0].finish_reason == "function_call"
        if is_func_call:
            completion = ""
            function_call = {
                "name": response.choices[0].message.function_call.name,
                "arguments": json.loads(response.choices[0].message.function_call.arguments),
            }
        else:
            completion = response.choices[0].message.content.strip()
            function_call = {}

        usage = response.usage

        meta = {
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
            "latency": latency.value,
        }
        return OpenAIResult(
            text=completion, inputs=model_inputs, provider=self, meta=meta, function_call=function_call
        )

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> OpenAIResult:
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency() as latency:
            response: Any = await self.client.chat.completions.create(
                model=self.model, **model_inputs  # type: ignore
            )

        completion = response.choices[0].message.content.strip()
        usage = response.usage

        meta = {
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
            "latency": latency.value,
        }
        return OpenAIResult(
            text=completion,
            inputs=model_inputs,
            provider=self,
            meta=meta,
        )


@ProviderRegistry.add
class Gpt35_Turbo(OpenAI):
    model: str = "gpt-3.5-turbo"


@ProviderRegistry.add
class Gpt35_Turbo_16k(OpenAI):
    model: str = "gpt-3.5-turbo-16k"


@ProviderRegistry.add
class Gpt35_Turbo_1106(OpenAI):
    model: str = "gpt-3.5-turbo-1106"


@ProviderRegistry.add
class Gpt4(OpenAI):
    model: str = "gpt-4"


@ProviderRegistry.add
class Gpt4_32k(OpenAI):
    model: str = "gpt-4-32k"


@ProviderRegistry.add
class Gpt4_1106(OpenAI):
    model: str = "gpt-4-1106-preview"
