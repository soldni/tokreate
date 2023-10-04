import json
from typing import Any, List, Optional

import aiohttp
import msgspec
import openai

from .base import BaseProvider, ProviderMessage, ProviderRegistry, ProviderResult


class OpenAIResult(ProviderResult):
    function_call: dict = msgspec.field(default_factory=dict)


class OpenAI(BaseProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        openai.api_key = api_key or openai.api_key
        openai.api_type = api_type or openai.api_type
        openai.api_base = api_base or openai.api_base
        openai.api_version = api_version or openai.api_version

    def _prepare_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        **kwargs,
    ) -> dict:
        messages = [ProviderMessage(role="user", content=prompt)]

        if history:
            messages = [*[ProviderMessage(**m) for m in history], ProviderMessage(role="user", content=prompt)]

        if system_message:
            messages = [*messages, ProviderMessage(role="system", content=system_message)]

        model_inputs = {
            "messages": [msgspec.structs.asdict(m) for m in messages],
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
        temperature: float = 0,
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
            response: Any = openai.ChatCompletion.create(model=self.model, **model_inputs)

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
            "tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": usage["completion_tokens"],
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
        temperature: float = 0,
        max_tokens: int = 300,
        aiosession: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ) -> OpenAIResult:
        if aiosession is not None:
            openai.aiosession.set(aiosession)  # type: ignore

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        with self.track_latency() as latency:
            response: Any = await openai.ChatCompletion.acreate(model=self.model, **model_inputs)  # type: ignore

        completion = response.choices[0].message.content.strip()
        usage = response.usage

        meta = {
            "tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": usage["completion_tokens"],
            "latency": latency.value,
        }
        return OpenAIResult(
            text=completion,
            inputs=model_inputs,
            provider=self,
            meta=meta,
        )


@ProviderRegistry
class Gpt35Turbo(OpenAI):
    model: str = "gpt-3.5-turbo"


@ProviderRegistry
class Gpt35Turbo16k(OpenAI):
    model: str = "gpt-3.5-turbo-16k"


@ProviderRegistry
class Gpt4(OpenAI):
    model: str = "gpt-4"


@ProviderRegistry
class Gpt432k(OpenAI):
    model: str = "gpt-4-32k"
