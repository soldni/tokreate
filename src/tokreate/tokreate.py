import copy
from abc import abstractmethod
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from jinja2 import Template
from msgspec import Struct, field

from .providers import ProviderRegistry, ProviderResult
from .utils import import_function_from_string


class Turn(Struct):
    role: Literal["user", "assistant"]
    content: Any
    state: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def as_input(self) -> dict:
        return {"role": self.role, "content": str(self.content)}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.role}>>> {str(self.content)}"


class BaseAction:
    @property
    def next(self) -> Optional["BaseAction"]:
        return getattr(self, "__next__", None)

    @next.setter
    def next(self, value: "BaseAction"):
        self.__next__ = value

    @abstractmethod
    def step(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        raise NotImplementedError()

    async def astep(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        # by default, just run the sync version
        return self.step(history=history, **kwargs)

    def run(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        output = self.step(history=history, **kwargs)
        if self.next is not None:
            output = self.next.run(history=output)
        return output

    async def arun(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        output = await self.astep(history=history, **kwargs)
        if self.next is not None:
            output = await self.next.arun(history=output)
        return output

    def __add__(self, other: "BaseAction") -> "BaseAction":
        self_copy = copy.copy(self)
        self_copy.next = (self_copy.next + other) if self_copy.next else other
        return self_copy

    def __radd__(self, other: "BaseAction") -> "BaseAction":
        return other + self


class ParseAction(BaseAction):
    def __init__(self, parser: Union[Callable[[str], dict], str]):
        if isinstance(parser, str):
            parser = import_function_from_string(parser)
        self.parser = parser

    def step(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        if history is None:
            # nothing to parse
            return []

        *history, raw_turn = history
        parsed_content = self.parser(raw_turn.content)
        parsed_turn = Turn(
            role=raw_turn.role,
            content=parsed_content,
            meta=raw_turn.meta,
            state={**raw_turn.state, "raw_content": raw_turn.content},
        )
        return [*history, parsed_turn]


class ClearAction(BaseAction):
    def step(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        return [history[-1]] if history else []


class CallAction(BaseAction):
    def __init__(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        parameters: Optional[dict] = None,
    ):
        self.prompt_template = Template(source=prompt)
        self.system_template = Template(source=system) if system else None
        self.model_function = ProviderRegistry.get(model=model)
        self.parameters = parameters or {}

    def _get_state(self, history: Optional[List[Turn]] = None, **kwargs) -> dict:
        state = {**(history[-1].state if history else {}), **kwargs}
        return state

    def _make_prompts(self, state: dict, history: Optional[List[Turn]] = None) -> Tuple[str, Optional[str]]:
        history = history or []
        message = self.prompt_template.render(history=history, **state)
        system = self.system_template.render(history=history, **state) if self.system_template else None
        return message, system

    def _make_user_turn(self, message: str, state: dict) -> Turn:
        return Turn(role="user", content=message, state=state)

    def _make_system_turn(self, output: Any, state: dict) -> Turn:
        assert isinstance(output, ProviderResult), f"Unexpected result type: {type(output)}"
        return Turn(role="assistant", content=output.text, meta={**output.meta, **output.inputs}, state=state)

    def step(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        state = self._get_state(history=history, **kwargs)
        message, system = self._make_prompts(state=state, history=history)
        user_turn = self._make_user_turn(message, state)
        result = self.model_function.complete(
            prompt=message,
            system_message=system,
            history=[turn.as_input() for turn in history] if history else None,
            **self.parameters,
        )
        system_turn = self._make_system_turn(result, state=state)
        return [*(history or []), user_turn, system_turn]

    async def astep(self, history: Optional[List[Turn]] = None, **kwargs) -> List[Turn]:
        state = self._get_state(history=history, **kwargs)
        message, system = self._make_prompts(state=state, history=history)
        user_turn = self._make_user_turn(message=message, state=state)
        result = await self.model_function.acomplete(
            prompt=message,
            system_message=system,
            history=[turn.as_input() for turn in history] if history else None,
            **self.parameters,
        )
        system_turn = self._make_system_turn(result, state=state)
        return [*(history or []), user_turn, system_turn]
