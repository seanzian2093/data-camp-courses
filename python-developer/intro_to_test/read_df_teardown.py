import pytest
import pandas as pd


@pytest.fixture
def data():
    df = pd.read_csv("data/domino.csv")
    yield df
    # Customized way to clear the content of the df
    df.drop(df.index, inplace=True)
    del df


def test_type(data):
    assert type(data) == pd.DataFrame, "Not a DataFrame"


def test_shape(data):
    assert data.shape[0] > 0, "Shape[0] is incorrect"
    assert data.shape[1] > 0, "Shape[1] is incorrect"
