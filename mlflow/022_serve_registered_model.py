import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Load the Production stage of Insurance model using scikit-learn flavor
model = mlflow.sklearn.load_model("models:/Insurance/Production")

# Run prediction on our test data
predictions = model.predict(X_test)
print(predictions)

# Serve a registered model from the command line
# mlflow models serve -m "models:/Insurance/Production" -p 1234
