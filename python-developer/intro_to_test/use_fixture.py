import pytest


# Use fixture decorator to define a fixture
@pytest.fixture
def list_length():
    return 10


# Use fixture decorator to define a fixture
@pytest.fixture
def prepare_data(list_length):
    return [i for i in range(list_length)]


# Use the fixture in a test function by passing the fixture as an argument
def test_element(prepare_data):
    assert 9 in prepare_data, "Element not found"
    assert 10 not in prepare_data, "Element found"


@pytest.fixture
def init_list():
    return []


# Use the autouse parameter to automatically use the fixture in all tests
@pytest.fixture(autouse=True)
def add_numbers_to_list(init_list):
    init_list.extend([i for i in range(10)])


def test_element2(init_list):
    assert 9 in init_list, "Element not found"
    assert 10 not in init_list, "Element found"
