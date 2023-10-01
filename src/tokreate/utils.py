import importlib
from typing import Callable


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
