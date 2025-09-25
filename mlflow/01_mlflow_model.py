import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Training Data
df = pd.read_csv("data/50_Startups.csv")

# Convert State to integer using factorize and then to float to avoid a wanring
df["State"] = pd.factorize(df["State"])[0].astype(float)

X = df[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = df[["Profit"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0
)

# Set the experiment to "Sklearn Model"
# mlflow.create_experiment("Sklearn Model")
mlflow.set_experiment("Sklearn Model")

# Set Auto logging for Scikit-learn flavor - requires xz package at sys level
# disable logging input examples to avoid warning
# mlflow.sklearn.autolog(
#     log_input_examples=False,
#     log_model_signatures=True,
# )
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# Mannually log model to MLflow Tracking - must disable autolog first
with mlflow.start_run():
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Log metrics manually
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    # Create an input example from your training data
    input_example = X_train.iloc[:1]
    # `name` argument is for model tracking service, not a path or folder
    # mlflow.sklearn.log_model(lr, name="lr_tracking", input_example=input_example)
    mlflow.sklearn.log_model(lr, name="lr_tracking2", input_example=input_example)

# Get a prediction from test data
print(lr.predict(X_test.iloc[[5]]))

# Save model to local filesystem, not to MLflow Tracking
mlflow.sklearn.save_model(lr, "lr_local_v1")

# Get the last run
run = mlflow.last_active_run()

# Get the run_id of the above run
run_id = run.info.run_id
print(run_id)

# Check what's actually in the artifacts directory
base_artifact_path = run.info.artifact_uri.replace("file://", "")
print(f"Looking for artifacts at: {base_artifact_path}")
if os.path.exists(base_artifact_path):
    print(f"Contents of artifact directory: {os.listdir(base_artifact_path)}")
else:
    print("Artifact directory doesn't exist!")

# Here `lr_tracking` is the outputs tag, not a folder or path, the folder name is `../outputs`
outputs_uri = f"{os.path.dirname(base_artifact_path)}/outputs"
print(f"Looking for outputs at: {outputs_uri}")
if os.path.exists(outputs_uri):
    # `outputs` contains all logged models meta information/meta.yml
    # detailed model files are in `experiment_id/models`
    print(f"Contents of outputs directory: {os.listdir(outputs_uri)}")
else:
    print("Outputs directory doesn't exist!")

# Load model from MLflow Tracking - here `lr_tracking` is the outputs tag, not a folder or path
# model = mlflow.sklearn.load_model(f"runs:/{run_id}/lr_tracking")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lr_tracking2")
print(model)
