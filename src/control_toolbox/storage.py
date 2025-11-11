from __future__ import annotations
from pydantic import BaseModel, Field
from typing import TypeVar, Type, Generic, Dict, Optional, Callable, Tuple, TypeAlias
import uuid
from typing import Protocol
from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass
T = TypeVar('T', bound=BaseModel)

import threading
from logging import getLogger
logger = getLogger(__name__)

class IDataStorage(Generic[T], Protocol):
    def store(self, data: T, id: Optional[str]) -> str: ...
    def retrieve(self, id: str) -> T: ...
    def has(self, id: str) -> bool: ...
    def delete(self, id: str) -> bool: ...


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




ReprT = TypeVar('ReprT', bound=BaseModel, covariant=True)

@dataclass
class StoredRepresentation(Generic[T, ReprT]):
    repr_id: Optional[str] = Field(..., description="ID of the representation. Used to retrieve the data from the storage.")
    kind: Type[T] = Field(..., description="Type of the data that is represented.")
    content: ReprT = Field(..., description="Teaser of the representation. Used to display the representation in a UI.")

class IConverter(Generic[ReprT, T], Protocol):
    def convert(self, data: T, id: Optional[str]) -> StoredRepresentation[T, ReprT]: ...

Converter:TypeAlias = Callable[[T, Optional[str]], StoredRepresentation[T, ReprT]]

class ReprStore(Generic[T, ReprT]):
    
    def __init__(self, storage: IDataStorage[T], stored_type: Type[T],
                 repr_type: Type[ReprT], convert_fn: Converter[T, ReprT]):
        self.storage: IDataStorage[T] = storage
        self._convert: Converter[T, ReprT] = convert_fn
        self.stored_type:Type[T] = stored_type
        self.repr_type: Type[ReprT] = repr_type

    def recreate(self, repr: StoredRepresentation[ReprT, T]|str) -> T:
        if isinstance(repr, str): return self.storage.retrieve(repr)
        if repr.repr_id is None:
            raise ValueError("The repr_id is not provided in the to_repr method or in this call.")
        return self.storage.retrieve(repr.repr_id)

    def convert(self, data: T, id: Optional[str] = None) -> StoredRepresentation[T, ReprT]:
        sr: StoredRepresentation[T, ReprT] = self._convert(data, id)
        _id = self.storage.store(data, sr.repr_id or id)
        sr.repr_id = _id
        return sr
        