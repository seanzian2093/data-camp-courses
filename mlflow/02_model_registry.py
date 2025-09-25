import mlflow

# Create an instance of MLflow Client Class named client
client = mlflow.MlflowClient()

# Create new model- once only
# client.create_registered_model("Insurance")

# print(client)

# Insurance filter string - LIKE ILIKE, =, !=
insurance_filter_string = "name LIKE 'Insurance'"

# Search for Insurance models
print(client.search_registered_models(filter_string=insurance_filter_string))

# Not Insurance filter string
not_insurance_filter_string = "name != 'Insurance'"

# Search for non Insurance models
print(client.search_registered_models(filter_string=not_insurance_filter_string))

# Register the first (2022) model - from a local file system directory
mlflow.register_model("./lg_local_v1", "Insurance")

# Register the second (2023) model - a logged model, using model tracking name
run_id = "7f38d85dff1e4074b1eb668cdbea89e6"
model_tracking_name = "sk_lr_model"
mlflow.register_model(f"runs:/{run_id}/{model_tracking_name}", "Insurance")
