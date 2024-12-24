import pytest


@pytest.fixture
def prepare_data():
    data = [i for i in range(10)]
    # Use yield to return the data but also to run the teardown code after the test
    yield data
    # Clear the content after the test
    data.clear()
    # Delete the variable after the test
    del data


def test_element(prepare_data):
    assert 9 in prepare_data, "Element not found"
    assert 10 not in prepare_data, "Element found"
