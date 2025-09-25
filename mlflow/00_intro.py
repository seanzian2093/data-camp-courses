import mlflow

# Create new experiment - once only
# mlflow.create_experiment("Unicorn Model")
# mlflow.create_experiment("LR Experiment")

# Tag new experiment
mlflow.set_experiment_tag("version", "1.0")

# Set the active experiment
# mlflow.set_experiment("Unicorn Model")
mlflow.set_experiment("LR Experiment")

# Start a run
run = mlflow.start_run()

print(run.info)

# Suppose we have done some training code in `train.py`
# model = LinearRegression(n_jobs=1)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# r2_score = r2_score(y_test, y_pred)
# r2_score = 0.95
r2_score = 0.88
# r2_score = 0.65

# Log the metric r2_score as "r2_score"
mlflow.log_metric("r2_score", r2_score)

# Log parameter n_jobs as "n_jobs"
mlflow.log_param("n_jobs", 1)

# Log the training code
mlflow.log_artifact("train.py")

# Create a filter string for R-squared score
r_squared_filter = "metrics.r2_score > .70"

# Search runs
r = mlflow.search_runs(
    experiment_names=["Unicorn Model", "LR Experiment"],
    filter_string=r_squared_filter,
    order_by=["metrics.r2_score desc"],
)

print(r)
# End a run
mlflow.end_run()
