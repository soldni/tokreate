import time
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar

USER_ROLE: str = "user"
SYSTEM_ROLE: str = "system"
ASSISTANT_ROLE: str = "assistant"


@dataclass(frozen=True, eq=True)
class ProviderMessage:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in (USER_ROLE, SYSTEM_ROLE, ASSISTANT_ROLE):
            raise ValueError(f"Role {self.role} not supported.")

    def asdict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, eq=True)
class ProviderResult:
    text: str
    inputs: Dict[str, Any]
    provider: "BaseProvider"
    meta: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class Latency:
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


T = TypeVar("T", bound="BaseProvider")


class ProviderRegistry:
    __registry__: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def add(cls, provider: Type[T]) -> Type[T]:
        name = getattr(provider, "model", None) or getattr(provider, "engine", None)
        assert name is not None, f"Provider {provider} has no model or engine attribute."
        cls.__registry__[name] = provider
        return provider

    @classmethod
    def get(cls, model: str, *args, **kwargs) -> BaseProvider:
        return cls.__registry__[model](*args, **kwargs)

    @classmethod
    def all(cls) -> List[str]:
        return list(cls.__registry__.keys())
