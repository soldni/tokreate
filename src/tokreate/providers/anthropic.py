import os
from typing import List, Optional

import anthropic

from .base import ASSISTANT_ROLE, USER_ROLE, BaseProvider, ProviderMessage, ProviderRegistry, ProviderResult


class AntrhopicMessage(ProviderMessage):
    def to_anthropic(self) -> str:
        if self.role == USER_ROLE:
            return f"{anthropic.HUMAN_PROMPT}{self.content}"
        elif self.role == ASSISTANT_ROLE:
            return f"{anthropic.AI_PROMPT}{self.content}"
        else:
            raise ValueError(f"Role {self.role} not supported.")


class Anthropic(BaseProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _prepare_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        ai_prompt: str = "",
        max_tokens_to_sample: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> dict:
        max_tokens_to_sample = max_tokens_to_sample or max_tokens

        messages = [
            *[AntrhopicMessage(**m) for m in (history or [])],
            AntrhopicMessage(role=USER_ROLE, content=prompt),
            AntrhopicMessage(role=ASSISTANT_ROLE, content=ai_prompt),
        ]
        formatted_prompt = "".join([m.to_anthropic() for m in messages])

        if stop_sequences is None:
            stop_sequences = [anthropic.HUMAN_PROMPT]

        model_inputs = {
            "prompt": formatted_prompt,
            "temperature": temperature,
            "max_tokens_to_sample": max_tokens_to_sample,
            "stop_sequences": stop_sequences,
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
        ai_prompt: str = "",
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ProviderResult:
        if system_message is not None:
            raise ValueError(f"system_message is not supported in {self.model}.")

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            **kwargs,
        )

        with self.track_latency() as latency:
            response = self.client.completions.create(model=self.model, **model_inputs)

        completion = response.completion.strip()

        meta = {"latency": latency.value}
        return ProviderResult(text=completion, inputs=model_inputs, provider=self, meta=meta)

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        ai_prompt: str = "",
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ProviderResult:
        if system_message is not None:
            raise ValueError(f"system_message is not supported in {self.model}.")

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            **kwargs,
        )

        with self.track_latency() as latency:
            response = await self.async_client.completions.create(model=self.model, **model_inputs)

        completion = response.completion.strip()

        meta = {"latency": latency.value}
        return ProviderResult(text=completion, inputs=model_inputs, provider=self, meta=meta)


@ProviderRegistry
class ClaudeInstantV11(Anthropic):
    model: str = "claude-instant-v1.1"


@ProviderRegistry
class ClaudeInstantV1(Anthropic):
    model: str = "claude-instant-v1"


@ProviderRegistry
class ClaudeV1(Anthropic):
    model: str = "claude-v1"


@ProviderRegistry
class ClaudeV1100k(Anthropic):
    model: str = "claude-v1-100k"


@ProviderRegistry
class ClaudeInstant1(Anthropic):
    model: str = "claude-instant-1"


@ProviderRegistry
class Claude2(Anthropic):
    model: str = "claude-2"
