from pydantic import BaseModel
from typing import TypeVar, Type, Generic, Dict, Optional
import uuid
from typing import Protocol
T = TypeVar('T', bound=BaseModel)
import threading

class IDataStorage(Generic[T], Protocol):
    def store(self, data: T, id:str) -> str: ...
    def retrieve(self, id: str) -> T: ...
    def has(self, id: str) -> bool: ...
    def delete(self, id: str) -> bool: ...

from collections.abc import MutableMapping, Iterator


class InMemoryDataStorage(Generic[T], MutableMapping[str, T]):
    def __init__(self, model_type: Type[T] = BaseModel, lock: Optional[threading.RLock] = None):
        self._type = model_type
        self._data: dict[str, T] = {}
        # RLock allows re-entrant locking if your domain methods call mapping methods
        self._lock = lock or threading.RLock()

    # --- Mapping protocol (source of truth) ---
    def __getitem__(self, id: str) -> T:
        with self._lock:
            return self._data[id]  # KeyError if missing (dict semantics)

    def __setitem__(self, id: str, data: T) -> None:
        # Optional runtime type enforcement:
        if not isinstance(data, self._type):
            raise TypeError(f"Expected {self._type.__name__}, got {type(data).__name__}")
        with self._lock:
            self._data[id] = data

    def __delitem__(self, id: str) -> None:
        with self._lock:
            del self._data[id]  # KeyError if missing

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            # snapshot keys to avoid "dictionary changed size during iteration"
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # --- Domain methods (delegate to mapping) ---
    def store(self, data: T, id: Optional[str] = None) -> str:
        if id is None:
            id = str(uuid.uuid4())
        # delegate to mapping (which already locks & validates)
        self[id] = data
        return id

    def retrieve(self, id: str) -> T:
        return self[id]

    def has(self, id: str) -> bool:
        with self._lock:
            return id in self._data

    def delete(self, id: str) -> bool:
        with self._lock:
            return self._data.pop(id, None) is not None