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

lr_class = LogisticRegression(max_iter=500)
lr_class.fit(X_train, y_train)

# Set experiment name
mlflow.set_experiment("Sklearn Logistic Model")

# Eval Data
eval_data = X_test
eval_data["sex"] = y_test
print(eval_data.dtypes)


# Log the lr_class model using Scikit-Learn Flavor
model_tracking_name = "sk_lr_model"

# Run artifacts will not be flushed untill a run is stopped so better using a `with` block
with mlflow.start_run():
    mlflow.sklearn.log_model(
        lr_class, name=model_tracking_name, input_example=X_train.iloc[:1]
    )

    # Get run id
    run = mlflow.last_active_run()
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Evaluate the logged model with eval_data data
    eval_res = mlflow.evaluate(
        f"runs:/{run_id}/{model_tracking_name}",
        data=eval_data,
        targets="sex",
        model_type="classifier",
    )
    print(eval_res)
