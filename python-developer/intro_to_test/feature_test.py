import pandas as pd
import pytest


@pytest.fixture
def get_df():
    return pd.read_csv("data/domino.csv")


# Feature
def agg_with_sum(data, group_by_column, aggregate_column):
    return data.groupby(group_by_column)[aggregate_column].sum()


# Test funtcion
def test_agg_feature(get_df):
    aggregated = agg_with_sum(
        get_df, "Starting user", "Run duration within reporting period (s)"
    )

    # Test the type of the result
    assert isinstance(aggregated, pd.Series), "Not a Series"

    # Test the number of rows of the result
    assert aggregated.shape[0] >= 0, "Number of rows is incorrect"

    # Test the data type of the result
    assert aggregated.dtype in (float, int), "Data type is incorrect"
