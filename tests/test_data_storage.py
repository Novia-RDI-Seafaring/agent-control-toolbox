from control_toolbox.storage import InMemoryDataStorage
from dataclasses import dataclass
import pytest

@dataclass
class SomeData:
    name: str
    value: float

@pytest.fixture(scope="session")
def storage() -> InMemoryDataStorage[SomeData]:
    return InMemoryDataStorage[SomeData]()

def test_add_data(storage):
    data = SomeData(name="test", value=1.0)
    id = storage.store(data)
    assert storage.has(id), f"Data with id {id} should be in storage"
    assert storage.retrieve(id) == data, f"Data with id {id} should be {data}"

    assert not storage.has("foo"), f"Data with id foo should not be in storage"

def test_retrieve_data(storage):
    data = SomeData(name="FooBar", value=99.0)
    id = "1234567890"
    
    assert not storage.has(id), f"Data with id {id} should not be in storage"    
    id2 = storage.store(data, id)
    assert id2 == id, f"Data with id {id} should be stored with the same id"
    assert storage.has(id2), f"Data with id {id2} should be in storage"
    assert storage.has(id), f"Data with id {id} should be in storage"
    data2 = storage.retrieve(id)
    assert data2 == data, f"Data with id {id} should be {data}"

def test_delete_data(storage):
    data = SomeData(name="BarBaz", value=22.0)
    id = storage.store(data)
    assert storage.has(id), f"Data with id {id} should be in storage"
    storage.delete(id)
    assert not storage.has(id), f"Data with id {id} should not be in storage"