import mlflow

# Create an instance of MLflow Client Class named client
client = mlflow.MlflowClient()

# Transition version 2 of Insurance model to stable stage
client.transition_model_version_stage(name="Insurance", version=2, stage="Production")

# Transition version 3 of Insurance model to testing stage
client.transition_model_version_stage(name="Insurance", version=3, stage="Staging")

# Transition version 1 of Insurance model to archive stage
client.transition_model_version_stage(name="Insurance", version=1, stage="Archived")
