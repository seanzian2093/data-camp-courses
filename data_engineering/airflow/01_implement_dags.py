import json
import requests
from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator


# Airflow DAGs can contain many operators, each performing their defined tasks
with DAG(
    dag_id="data_camp_course_bash", default_args={"start_date": "2024-01-01"}
) as analytics_dag:
    # Define the BashOperator
    cleanup = BashOperator(task_id="cleanup_task", bash_command="cleanup.sh")

    # Define a second operator to run the `consolidate_data.sh` script
    consolidate = BashOperator(
        task_id="consolidate_task", bash_command="consolidate_data.sh"
    )

    # Define a final operator to execute the `push_data.sh` script
    push_data = BashOperator(task_id="pushdata_task", bash_command="push_data.sh")

    # Define a new pull_sales task
    pull_sales = BashOperator(
        task_id="pullsales_task",
        bash_command="wget https://salestracking/latestinfo?json",
    )

    # Set pull_sales to run prior to cleanup
    pull_sales >> cleanup

    # Configure consolidate to run after cleanup
    consolidate << cleanup

    # Set push_data to run last
    consolidate >> push_data


# Python Operator example
default_args = {
    "owner": "sales_eng",
    "start_date": datetime(2023, 2, 15),
}

process_sales_dag = DAG(
    dag_id="data_camp_course_python", default_args=default_args, schedule="@monthly"
)


def pull_file(URL, savepath):
    r = requests.get(URL)
    with open(savepath, "wb") as f:
        f.write(r.content)
    # Use the print method for logging
    print(f"File pulled from {URL} and saved to {savepath}")


# Create the task
pull_file_task = PythonOperator(
    task_id="pull_file",
    # Add the callable
    python_callable=pull_file,
    # Define the arguments
    op_kwargs={"URL": "http://dataserver/sales.json", "savepath": "latestsales.json"},
    dag=process_sales_dag,
)


# Add another Python task
def parse_file(inputfile, outputfile):
    with open(inputfile) as infile:
        data = json.load(infile)
        with open(outputfile, "w") as outfile:
            json.dump(data, outfile)


parse_file_task = PythonOperator(
    task_id="parse_file",
    # Set the function to call
    python_callable=parse_file,
    # Add the arguments
    op_kwargs={"inputfile": "latestsales.json", "outputfile": "parsedfile.json"},
    dag=process_sales_dag,
)


# Define the task
email_manager_task = EmailOperator(
    task_id="email_manager",
    to="manager@datacamp.com",
    subject="Latest sales JSON",
    html_content="Attached is the latest sales JSON file as requested.",
    files="parsedfile.json",
    dag=process_sales_dag,
)

# Set the order of tasks
pull_file_task >> parse_file_task >> email_manager_task
