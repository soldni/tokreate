from typing import Optional

import llms

from .logging import get_logger


class PatchProviders:
    __singleton: Optional["PatchProviders"] = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            cls.__singleton = super().__new__(cls)
        return cls.__singleton

    def __init__(self, *args, **kwargs):
        logger = get_logger(self.__class__.__name__)

        llms.llms.OpenAIProvider.MODEL_INFO.setdefault(
            "gpt-3.5-turbo-16k",
            {"prompt": 3.0, "completion": 4.0, "token_limit": 16_384},
        )
        logger.info("Added gpt-3.5-turbo-16k to llms.OpenAIProvider.MODEL_INFO")
