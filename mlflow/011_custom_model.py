import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
df = pd.read_csv(
    "data/insurance.csv",
    dtype={
        "age": "float64",
        "bmi": "float64",
        "children": "float",
        "charges": "float64",
        "smoker": "object",
        "sex": "object",
    },
)

# Convert State to integer using factorize and then to float to avoid a wanring
df["smoker"] = pd.factorize(df["smoker"])[0].astype(float)
df["sex"] = pd.factorize(df["sex"])[0].astype(float)

X = df[["age", "bmi", "children", "smoker", "charges"]]
y = df["sex"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0
)


# Create Python Class
class CustomPredict(mlflow.pyfunc.PythonModel):
    # Set method for loading model
    def load_context(self, context):
        # Load the model from the artifacts directory
        self.model = mlflow.sklearn.load_model(context.artifacts["lr_pyfunc"])

    # Set method for custom inference
    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        decoded_predictions = []
        for prediction in predictions:
            if prediction == 0:
                decoded_predictions.append("female")
            else:
                decoded_predictions.append("male")
        return decoded_predictions


mlflow.set_experiment("Custom Model")

# Log the pyfunc model
with mlflow.start_run():
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    # Save model to local filesystem first
    model_dir = "lg_local_v1"
    model_tracking_name = "lr_pyfunc"
    mlflow.sklearn.save_model(lr, model_dir)

    input_example = X_train.iloc[:1]
    mlflow.pyfunc.log_model(
        name=model_tracking_name,
        # Set model to use CustomPredict Class
        python_model=CustomPredict(),
        input_example=input_example,
        # Include the saved model as an artifact so that we can load it using `runs:/<run_id>/lr_pyfunc`
        artifacts={model_tracking_name: model_dir},
        signature=mlflow.models.infer_signature(X_train, lr.predict(X_train)),
    )

run = mlflow.last_active_run()
run_id = run.info.run_id
print(f"Run ID: {run_id}")

# Load the model in python_function format
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_tracking_name}")
print(loaded_model)
