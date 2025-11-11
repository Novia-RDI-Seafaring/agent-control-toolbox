from pydantic import BaseModel
from typing import TypeVar, Type, Generic, Dict
import uuid
from typing import Protocol
T = TypeVar('T', bound=BaseModel)

class DataStorage(Generic[T], Protocol):
    def store(self, data: T, id:str=str(uuid.uuid4())) -> str:
        ...

    def retrieve(self, id: str) -> T:
        ...

    def has(self, id: str) -> bool:
        ...

    def delete(self, id: str) -> bool:
        ...

class InMemoryDataStorage(Generic[T], DataStorage[T]):
    def __init__(self):
        self.data: Dict[str, T] = {}

    def store(self, data: T, id:str=str(uuid.uuid4())) -> str:
        self.data[id] = data
        return id

    def has(self, id: str) -> bool:
        return id in self.data

    def retrieve(self, id: str) -> T:
        return self.data[id]
    
    def delete(self, id: str) -> bool:
        if id in self.data:
            del self.data[id]
            return True
        return False