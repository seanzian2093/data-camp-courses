import great_expectations as gx
import pandas as pd

dataframe = pd.read_csv("data/weather_new.csv")
context = gx.get_context()
data_source = context.data_sources.add_pandas(name="my_pandas_datasource")
data_asset = data_source.add_dataframe_asset(name="my_data_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe(
    name="my_batch_definition"
)
batch = batch_definition.get_batch(batch_parameters={"dataframe": dataframe})

# Row level expectations
# Establish missingness Expectation
expectation = gx.expectations.ExpectColumnValuesToNotBeNull(column="name")

validation_results = batch.validate(expect=expectation)
print(validation_results.success)

# Establish type Expectation
expectation = gx.expectations.ExpectColumnValuesToBeOfType(column="sku_id", type_="str")

validation_results = batch.validate(expect=expectation)
print(validation_results.success)

suite = context.suites.add(gx.ExpectationSuite(name="my_suite", suite_parameters={}))
# "colour" should be in the set "Khaki", "Purple", or "Grey"
colour_expectation = gx.expectations.ExpectColumnDistinctValuesToBeInSet(
    column="colour", value_set={"Khaki", "Purple", "Grey"}
)

# "seller_name" should have 7 to 10 distinct values
seller_expectation = gx.expectations.ExpectColumnUniqueValueCountToBeBetween(
    column="seller_name", min_value=7, max_value=10
)

# "link" should have all unique values
link_expectation = gx.expectations.ExpectColumnValuesToBeUnique(column="link")

# "review_count" should have a most common value in the set "0" or "100+"
review_count_expectation = gx.expectations.ExpectColumnMostCommonValueToBeInSet(
    column="review_count", value_set={"0", "100+"}
)

# Column median Expectation
col_median_expectation = gx.expectations.ExpectColumnMedianToBeBetween(
    column="star_rating",
    min_value=2,
    max_value=4,
)

# Column values increasing Expectation
col_values_increasing_expectation = gx.expectations.ExpectColumnValuesToBeIncreasing(
    column="star_rating"
)

# Column value lengths Expectation
expectation = gx.expectations.ExpectColumnValueLengthsToEqual(column="name", value=100)

# Establish Conditional Expectation
expectation = gx.expectations.ExpectColumnValuesToBeInSet(
    column="star_rating",
    value_set={0.0},
    condition_parser="pandas",
    row_condition="review_count==0",
)

# Create list of Expectations
expectations = [
    colour_expectation,
    seller_expectation,
    link_expectation,
    review_count_expectation,
    col_median_expectation,
]

# Add Expectations to Suite
for expectation in expectations:
    suite.add_expectation(expectation)

# Run Validation
validation_results = batch.validate(expect=suite)

# Print success status of Validation Results
print(validation_results.success)
