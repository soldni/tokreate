from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Type

from jinja2 import BaseLoader, Environment, Template, meta


class JinjaEnvironment:
    """A singleton for the jinja environment."""

    _env: Optional["Environment"] = None

    @classmethod
    def env(cls, loader: Optional[Type["BaseLoader"]] = None) -> "Environment":
        if cls._env is not None:
            return cls._env

        cls._env = Environment(loader=(loader or BaseLoader))  # pyright: ignore
        return cls._env

    @classmethod
    def from_string(cls, template: str, env_kwargs: Optional[dict] = None) -> "Template":
        return cls.env(**(env_kwargs or {})).from_string(template)

    @classmethod
    def find_undeclared_variables(cls, template: str) -> List[str]:
        """Find undeclared variables in a jinja template."""
        ast = cls.env().parse(template)
        return sorted(meta.find_undeclared_variables(ast))


@dataclass(frozen=True, eq=True, repr=True)
class Prompt:
    prompt: str

    @cached_property
    def template(self) -> Template:
        return JinjaEnvironment.from_string(self._get_prompt())

    def _get_prompt(self) -> str:
        if (prompt := getattr(self, "prompt", None)) is None:
            raise NotImplementedError(f"Prompt must be defined in {self.__class__.__name__}")
        return prompt

    def render(self, **kwargs: str) -> str:
        return self.template.render(**kwargs)

    @cached_property
    def variables(self) -> List[str]:
        return JinjaEnvironment.find_undeclared_variables(self._get_prompt())

    @classmethod
    def from_json(cls, __dict: dict) -> "Prompt":
        return cls(**__dict)

    def to_json(self) -> dict:
        return dict(prompt=self.prompt)


@dataclass(frozen=True, eq=True, repr=True)
class Turn:
    role: str
    content: str

    @classmethod
    def from_json(cls, __dict: dict) -> "Turn":
        return cls(**__dict)

    def to_json(self) -> dict:
        return dict(role=self.role, content=self.content)


class History(List[Turn]):
    """Collect a list of turns in a conversation."""

    def __init__(self, __list: Optional[List[Turn]] = None) -> None:
        super().__init__(__list or [])
        for item in self:
            self._check_type(item)

    def _check_type(self, __object: Turn):
        if not isinstance(__object, Turn):
            raise TypeError(f"History must be a list of Turn objects, got {type(__object)}")

    def append(self, __object: Turn) -> None:
        self._check_type(__object)
        return super().append(__object)

    def add(self, role: str, content: str) -> None:
        self.append(Turn(role=role, content=content))

    @classmethod
    def from_json(cls, __list: List[dict]) -> "History":
        return cls([Turn.from_json(item) for item in __list])

    def to_json(self) -> List[dict]:
        return [item.to_json() for item in self]
