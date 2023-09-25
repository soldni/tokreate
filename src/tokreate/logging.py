import os
import sys
from logging import INFO, Formatter, Handler, Logger, StreamHandler, getLogger
from typing import Optional, Union

import boto3
import watchtower
from logging_json import JSONFormatter


def get_client():
    region_name = os.environ.get("AWS_REGION", "us-west-2")
    return boto3.client("logs", region_name=region_name)


def get_formatter():
    fields = {
        "level_name": "levelname",
        "timestamp": "asctime",
        "modulename": "module",
        "functionname": "funcName",
    }
    return JSONFormatter(fields=fields)


def get_stream_handler(formatter: Formatter):
    stream_handler = StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    return stream_handler


def get_cloudwatch_handler(stream_name: str, formatter: Formatter, log_group: str, client=None) -> Handler:
    cloudwatch_handler = watchtower.CloudWatchLogHandler(
        log_group=log_group,
        stream_name=stream_name,
        boto3_client=client,
        send_interval=15,
    )
    cloudwatch_handler.setFormatter(formatter)
    return cloudwatch_handler


def get_logger(
    name: str,
    enable_all_logging: Union[str, bool] = os.getenv("ENABLE_ALL_LOGGING", "true"),
    cloudwatch_log_group: Optional[str] = os.getenv("CLOUDWATCH_LOG_GROUP"),
) -> Logger:
    logger = getLogger(name)
    logger.setLevel(INFO)

    if isinstance(enable_all_logging, str):
        enable_all_logging = enable_all_logging.lower().strip() in ["true", "1"]

    if not enable_all_logging:
        return logger

    formatter = get_formatter()
    stream_handler = get_stream_handler(formatter)
    logger.addHandler(stream_handler)

    if not cloudwatch_log_group:
        return logger

    try:
        client = get_client()
        cloudwatch_handler = get_cloudwatch_handler(
            stream_name=name, log_group=cloudwatch_log_group, formatter=formatter, client=client
        )
        logger.addHandler(cloudwatch_handler)
    except Exception:
        logger.error("Failed to connect to CloudWatch")

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    for msg in sys.argv[1:]:
        logger.info(msg)
