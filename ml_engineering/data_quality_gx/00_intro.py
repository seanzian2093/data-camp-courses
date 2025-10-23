import great_expectations as gx
import pandas as pd

dataframe = pd.read_csv("data/weather_new.csv")

# Create Data Context - primary entry point for Great Expectations
context = gx.get_context()

# Print Data Context metadata
# print(context)

# Create Data Source - interface to data system
data_source = context.data_sources.add_pandas(name="my_pandas_datasource")

# Create Data Asset - structured representation of data
data_asset = data_source.add_dataframe_asset(name="my_data_asset")

# Create Batch Definition
batch_definition = data_asset.add_batch_definition_whole_dataframe(
    name="my_batch_definition"
)

# Load data to a batch
batch = batch_definition.get_batch(batch_parameters={"dataframe": dataframe})

# Inspect the batch
# print(batch.head())
# print(batch.columns())

# Establish and evaluate an Expectation
row_count_expectation = gx.expectations.ExpectTableRowCountToBeBetween(
    min_value=50000, max_value=100000
)

# Validate the batch against the Expectation
# validation_results = batch.validate(expect=row_count_expectation)
# print(validation_results.success)
# print(validation_results.result)
# print(validation_results.describe())

# Create Expectation Suite and Validation Definition
suite = context.suites.add(gx.ExpectationSuite(name="my_suite", suite_parameters={}))

# Add Expectation to Suite - one Expectation can not belong to multiple Suites at same time
suite.add_expectation(expectation=row_count_expectation)

# Create and add to suite another Expectation
col_name_expectation = gx.expectations.ExpectColumnToExist(column="my_column_name")

suite.add_expectation(expectation=col_name_expectation)

# Update Expectation
col_name_expectation.column = "Location"

# Save Expectation
col_name_expectation.save()

# View Suite's Expectations
# print(suite.expectations)

# Validate Suite
# validation_results = batch.validate(expect=suite)

# Describe Validation Results
# print(validation_results.describe())

# Validate Expectation Suite
validation_definition = gx.ValidationDefinition(
    data=batch_definition, suite=suite, name="validation"
)
# validation_results = validation_definition.run(
#     batch_parameters={"dataframe": dataframe}
# )

# View success status of Validation Results
# print(validation_results.success)

# Create Checkpoint

checkpoint = gx.Checkpoint(
    name="my_checkpoint", validation_definitions=[validation_definition]
)

# Add Checkpoint to Context
context.validation_definitions.add(validation_definition)
context.checkpoints.add(checkpoint)

# Run Checkpoint
checkpoint_results = checkpoint.run(batch_parameters={"dataframe": dataframe})

# Print success status
print(checkpoint_results.success)

# Delete the Expectation
suite.delete_expectation(expectation=col_name_expectation)

# Save changes before run
suite.save()
