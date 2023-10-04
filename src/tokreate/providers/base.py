import time
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Type

from msgspec import Struct, field

USER_ROLE: str = "user"
SYSTEM_ROLE: str = "system"
ASSISTANT_ROLE: str = "assistant"


class ProviderMessage(Struct):
    role: str
    content: str

    def __post_init__(self):
        if self.role not in (USER_ROLE, SYSTEM_ROLE, ASSISTANT_ROLE):
            raise ValueError(f"Role {self.role} not supported.")


class ProviderResult(Struct):
    text: str
    inputs: Dict[str, Any]
    provider: "BaseProvider"
    meta: Dict[str, Any] = field(default_factory=dict)


class Latency(Struct):
    value: float = field(default=-1.0)


class BaseProvider:
    model: str

    @contextmanager
    def track_latency(self) -> Generator[Latency, None, None]:
        start = time.perf_counter()

        latency = Latency()
        try:
            yield latency
        finally:
            latency.value = round(time.perf_counter() - start, 2)

    @abstractmethod
    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = -1.0,
        max_tokens: int = -1,
        **kwargs,
    ) -> ProviderResult:
        raise NotImplementedError()

    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = -1.0,
        max_tokens: int = -1,
        **kwargs,
    ) -> ProviderResult:
        raise NotImplementedError()


class ProviderRegistry:
    __registry__: Dict[str, Type[BaseProvider]] = {}

    def __new__(cls, provider: Type[BaseProvider]):
        cls.__registry__[provider.model] = provider

    @classmethod
    def get(cls, model: str, *args, **kwargs) -> BaseProvider:
        return cls.__registry__[model](*args, **kwargs)

    @classmethod
    def all(cls) -> List[str]:
        return list(cls.__registry__.keys())
