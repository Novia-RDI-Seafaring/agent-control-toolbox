from control_toolbox.storage import InMemoryDataStorage, IDataStorage
from dataclasses import dataclass
import pytest
from typing import Literal, Optional, Generic, Type, TypeVar
import uuid

@dataclass
class SomeData:
    name: str
    value: float

@pytest.fixture(scope="session")
def storage() -> InMemoryDataStorage[SomeData]:
    return InMemoryDataStorage[SomeData](model_type=SomeData)

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


def test_protocol():

    @dataclass
    class RingOfPower:
        name: str
        destination: Literal["Elves", "Kings of Men", "Dwarves", "Sauron"]

        @property
        def id(self) -> str: return self.name.lower().replace(" ", "_")

    def keep_it_hidden_keep_it_safe(ring: RingOfPower,s: IDataStorage[RingOfPower]) -> str:
        return s.store(ring, ring.id)
    
    storage = InMemoryDataStorage[RingOfPower](model_type=RingOfPower)

    my_precious = RingOfPower(name="the One Ring to rule them All", destination="Sauron")
    assert my_precious.id == "the_one_ring_to_rule_them_all", f"Data with id the_one_ring_to_rule_them_all should be stored with the same id"
    assert not storage.has(my_precious.id), f"Data with id foo should not be in storage"

    id = keep_it_hidden_keep_it_safe(my_precious, storage)
    assert id == my_precious.id, f"Data with id {my_precious.id} should be stored with the same id"

    some_old_ring = storage.retrieve(my_precious.id)
    assert some_old_ring == my_precious, f"My Precious is LOST!!!"

    storage.delete(some_old_ring.id)
    assert not storage.has(my_precious.id), f"Some things that should not have been forgotten were lost..."

T = TypeVar("T")

def test_abstract_data_storage():
    
    @dataclass
    class Thought:
        name: str
        description: str

    class SingleItemDataStorage(Generic[T]):
        _item: Optional[Thought] = None
        _id: Optional[str] = None

        def store(self, data: SomeData, id:str=None) -> str:
            self._item = data
            self._id = id or str(uuid.uuid4())
            return self._id
        
        def delete(self) -> bool:
            self._item = None
            self._id = None
            return True
        
        def retrieve(self, id: str) -> SomeData:
            if self._id == id:
                return self._item
            raise ValueError(f"Data with id {id} not found")
        
        def has(self, id: str) -> bool:
            return self._id == id


    with pytest.raises(TypeError):
        storage = SingleItemDataStorage[SomeData](type=SomeData)

    storage = SingleItemDataStorage[Thought]()

    with pytest.raises(ValueError):
        storage.retrieve("foo")

    the_best_idea = Thought(name="Idea for Fusion Reactor Implementation", description="I finally figured it out....")
    id = storage.store(the_best_idea)
    assert id is not None
    assert storage.has(id), f"Data with id {id} should be in storage"
    _uid1 = uuid.uuid4()
    _uid2 = uuid.uuid4()
    assert _uid1 != _uid2, f"UUIDs should not be the same"

    my_latest_thought = storage.retrieve(id)
    assert my_latest_thought == the_best_idea, f"My best idea should be {the_best_idea}"
    assert my_latest_thought.name == "Idea for Fusion Reactor Implementation", f"My best idea should be 'Idea for Fusion Reactor Implementation'"

    daydream = Thought(name="What's for lunch?", description="I would really fancy some pizza...")
    id2 = storage.store(daydream)
    print(f"id2: {id2}")
    

    assert id2 != id, f"Data with id {id2} should be stored with a new id, not the same as {id}"
    assert storage.has(id2), f"Data with id {id2} should be in storage"
    assert not storage.has(id), f"Data with id {id} should not be in storage"
