import os
import loguru

_logger = loguru.logger


class NullLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def bind(self, **kwargs):
        return self


def setup_logger(exp_dir: str):
    global _logger

    if int(os.getenv("RANK", 0)) <= 0:
        _logger.add(
            os.path.join(exp_dir, "train.log"),
            level="DEBUG",
            colorize=False,
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
        )
    else:
        _logger = NullLogger()

    _logger.info(f"Experiment directory created at: {exp_dir}")
    return _logger


def get_logger():
    return _logger


__all__ = ["setup_logger", "get_logger"]
