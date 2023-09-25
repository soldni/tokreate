import copy
import json
from dataclasses import dataclass
from functools import cached_property
from hashlib import md5
from logging import Logger
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union, cast

from llms import LLMS, init
from llms.llms import Result, Results

from .logging import get_logger
from .prompts import History, Prompt
from .utils import import_function_from_string

T = TypeVar("T")


@dataclass(frozen=True, eq=True, repr=True)
class Output:
    text: str
    history: History
    metadata: dict
    call: dict

    @classmethod
    def from_json(cls, __dict: dict) -> "Output":
        return cls(
            text=__dict["text"],
            history=History.from_json(__dict["history"]),
            metadata=__dict["metadata"],
            call=__dict["call"],
        )

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "history": self.history.to_json(),
            "metadata": self.metadata,
            "call": self.call,
        }


@dataclass(frozen=True, eq=True, repr=True)
class Call(Generic[T]):
    model_id: str
    prompt: Prompt
    system_message: Optional[Prompt] = None
    model_parameters: Optional[dict] = None
    output_parser: Optional[str] = None

    @cached_property
    def logger(self) -> Logger:
        return get_logger(self.__class__.__name__)

    @cached_property
    def model(self) -> LLMS:
        model = init(self.model_id)
        self.logger.info(f"Loaded model {self.model_id}")
        return model

    @cached_property
    def parser(self) -> Optional[Callable[[str], T]]:
        func = import_function_from_string(self.output_parser) if self.output_parser else None
        if func is not None:
            self.logger.info(f"Loaded parser `{self.output_parser}`")
        else:
            self.logger.info(f"No parser defined for {self.__class__.__name__}")
        return func

    @classmethod
    def from_json(cls, __dict: dict) -> "Call":
        return cls(
            model_id=__dict["model_id"],
            prompt=Prompt.from_json(__dict["prompt"]),
            system_message=Prompt.from_json(__dict["system_message"]) if __dict.get("system_message") else None,
            model_parameters=__dict.get("model_parameters"),
        )

    @cached_property
    def signature(self) -> str:
        return md5(json.dumps(self.to_json(), sort_keys=True).encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return {
            "model": self.model_id,
            "prompt": self.prompt.to_json(),
            "system_message": self.system_message.to_json() if self.system_message else None,
            "model_parameters": self.model_parameters,
        }

    def _make_output(self, result: Result, history: Optional[History] = None, inplace: bool = False) -> Output:
        if history is None:
            history = History()
        elif not inplace:
            history = copy.copy(history)

        history.add(**result.model_inputs.pop("messages")[-1])
        return Output(
            text=result.text,
            history=history,
            metadata=result.meta,
            call=self.to_json(),
        )

    def _pre_run(self, **variables: str) -> Tuple[str, Union[str, None]]:
        prompt = self.prompt.render(**variables)
        system_message = self.system_message.render() if self.system_message else None
        return prompt, system_message

    def _post_run(
        self, result: Union[Result, Results], history: Optional[History] = None, inplace: bool = False
    ) -> Output:
        if not isinstance(result, Result):
            raise ValueError(f"Unexpected output type: {type(result)}")
        output = self._make_output(result=result, history=history)
        self.logger.info(f"Response: {output.text}")
        return output

    def run(self, history: Optional[History] = None, **variables: str) -> Output:
        """Main entry point for running a model."""
        prompt, system_message = self._pre_run(**variables)
        result = self.model.complete(
            prompt=prompt, history=history, system_message=system_message, **(self.model_parameters or {})
        )
        return self._post_run(result=result, history=history)

    async def arun(self, history: Optional[History] = None, **variables: str) -> Output:
        """Same as `run` but async."""
        prompt, system_message = self._pre_run(**variables)
        result = await self.model.acomplete(
            prompt=prompt, history=history, system_message=system_message, **(self.model_parameters or {})
        )
        return self._post_run(result=result, history=history)

    def parse(self, output: Output) -> T:
        """Parse the output of a model."""
        if self.parser is None:
            return cast(T, output.text)
        return self.parser(output.text)
