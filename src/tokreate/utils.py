import hashlib
import importlib
import inspect
import pickle
from typing import TYPE_CHECKING, Callable, Type, TypeVar

if TYPE_CHECKING:
    from .tokreate import BaseAction


def import_function_from_string(func_string: str) -> Callable:
    if "." not in func_string:
        func = globals().get(func_string)
        if func is None:
            raise ValueError(f'Function "{func_string}" not found in globals.')
        return func

    # Split the module and function name
    module_name, function_name = func_string.rsplit(".", 1)

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get the function from the module
    function = getattr(module, function_name)

    if function is None:
        raise ValueError(f'Function "{function_name}" not found in module "{module_name}".')

    return function


T = TypeVar("T")


def add_signature(cls: Type[T]) -> Type[T]:
    """Use the parameters to the __init__ method to generate a signature to assign to hash."""

    old_init = cls.__init__
    signature = inspect.signature(old_init)

    def init_and_sign(self: T, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        hash_str = hashlib.md5(pickle.dumps({k: v for k, v in sorted(bound.arguments.items())})).hexdigest()
        old_init(self, *args, **kwargs)

        setattr(self, "signature", hash_str)

    setattr(cls, "__init__", init_and_sign)

    return cls


def get_signature(action: "BaseAction") -> str:
    signature = 0
    while True:
        current_signature = getattr(action, "signature", None)
        if current_signature is None:
            raise ValueError(f"Action {action} has no signature.")
        signature ^= int(current_signature, 16)

        if (action := getattr(action, "next", None)) is None:  # type: ignore
            break

    return hex(signature)[2:]
