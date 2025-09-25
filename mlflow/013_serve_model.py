import subprocess
import requests

# requires virtualenv installed
# Local Filesystem
# mlflow models serve -m relative/path/to/local/model

# Run ID
# mlflow models serve -m runs:/<mlflow_run_id>/artifacts/model

# AWS S3
# mlflow models serve -m s3://my_bucket/path/to/model

# Basic usage
cmd_lst = ["mlflow", "models", "serve", "-m", "lg_local_v1"]
result = subprocess.run(cmd_lst, capture_output=True, text=True)
print(result.stdout)

input_json = {
    "dataframe_split": {
        "columns": ["age", "bmi", "children", "smoker", "charges"],
        "data": [[18.0, 28.215, 0.0, 1.0, 2200.83085]],
    }
}

# Check return code
if result.returncode == 0:
    print("Command succeeded")
else:
    print("Command failed")
