import functools
from typing import Callable, Any

from llm_benchmarker.events.handlers import EventHandler
from loguru import logger


def slot(shared_key: str):
    def decorator(func: Callable[[str], Any]):
        func._slot_decorated = True
        func.__dataset__ = shared_key
        handler = EventHandler()
        handler.subscribe(event_name=shared_key, slot_fn=func)
        logger.debug(f"Slot \"{func.__name__}\" subscribed to \"{shared_key}\".")
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.debug(f"Slot \"{func.__name__}\" get a signal event \"{shared_key}\"")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def signal(shared_key: str):
    def decorator(func: Callable[[Any], Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            local_path = func(*args, **kwargs)
            handler = EventHandler()
            results = handler.emit(shared_key, local_path)
            logger.debug(f"A signal emit event \"{shared_key}\"")
            return results
        return wrapper
    return decorator
