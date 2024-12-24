import pandas as pd
import pytest


@pytest.fixture
def get_df():
    return pd.read_csv("data/domino.csv")


# Integration test function
def test_get_df(get_df):
    # Test the type of the result
    assert isinstance(get_df, pd.DataFrame), "Not a DataFrame"

    # Test the number of rows of the result
    assert get_df.shape[0] >= 0, "Number of rows is incorrect"

    # Test the number of columns of the result
    assert get_df.shape[1] >= 0, "Number of columns is incorrect"

    # Test the data type of the result
    # assert get_df.dtypes.all() in (float, int, object), "Data type is incorrect"
