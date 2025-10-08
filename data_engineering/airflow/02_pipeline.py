from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "start_date": datetime(2023, 4, 15),
}

cleandata_dag = DAG("cleandata", default_args=default_args, schedule_interval="@daily")

# Modify the templated command to handle a second argument called filename.
templated_command = """
  bash cleandata.sh {{ ds_nodash }} {{ params.filename }}
"""

# A template to loop over a list of files
# filelist = [f'file{x}.txt' for x in range(30)]
# templated_command = """
#   <% for filename in params.filenames %>
#   bash cleandata.sh {{ ds_nodash }} {{ filename }};
#   <% endfor %>
# """

# Modify clean_task to pass the new argument
clean_task = BashOperator(
    task_id="cleandata_task",
    bash_command=templated_command,
    # params will be used to pass argument by key in the bash_command
    params={"filename": "salesdata.txt"},
    dag=cleandata_dag,
)

# Create a new BashOperator clean_task2
clean_task2 = BashOperator(
    task_id="cleandata_task2",
    bash_command=templated_command,
    params={"filename": "supportdata.txt"},
    # params={'filenames': filelist},
    dag=cleandata_dag,
)

# Set the operator dependencies
clean_task >> clean_task2
