from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

dag = DAG(
    "BranchingTest",
    default_args={"start_date": datetime(2023, 4, 15)},
    schedule_interval="@daily",
)


# Branching function returns the name of task to follow
def branch_test(**kwargs):
    if int(kwargs["ds_nodash"]) % 2 == 0:
        return "even_day_task"
    else:
        return "odd_day_task"


start_task = EmptyOperator(task_id="start_task", dag=dag)

branch_task = BranchPythonOperator(
    task_id="branch_task", provide_context=True, python_callable=branch_test, dag=dag
)

even_day_task = EmptyOperator(task_id="even_day_task", dag=dag)
even_day_task2 = EmptyOperator(task_id="even_day_task2", dag=dag)

odd_day_task = EmptyOperator(task_id="odd_day_task", dag=dag)
odd_day_task2 = EmptyOperator(task_id="odd_day_task2", dag=dag)

start_task >> branch_task >> even_day_task >> even_day_task2
branch_task >> odd_day_task >> odd_day_task2
