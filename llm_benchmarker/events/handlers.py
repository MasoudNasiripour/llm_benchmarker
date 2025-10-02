import functools

from typing import Callable, Any, Tuple
from loguru import logger


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


class EventHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._slots: dict[str, list] = {}
        return cls._instance

    def __init__(self):
        ...


    def load_slots_from_list(self, event_slots: list[Tuple[str, Callable]]):
        for event_name, slot_fn in event_slots:
            self.subscribe(event_name, slot_fn)

    def subscribe(self, event_name: str, slot_fn: Callable[[Any], Any]):
        if event_name not in self._slots.keys():
            self._slots[event_name] = []
        self._slots[event_name].append(slot_fn)

    def unsubscribe(self, event_name: str, slot_fn: Callable[[], Any]):
        if event_name in self._slots.keys():
            self._slots[event_name].remove(slot_fn)

    def emit(self, event_name: str, *args, **kwargs) -> Any:
        results = {
        }
        if event_name in self._slots:
            for slot in self._slots[event_name]:
                results[event_name] = slot(*args, **kwargs)
        return results
