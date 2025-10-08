from datetime import datetime, timedelta
from airflow import DAG

# Update the scheduling arguments as defined
default_args = {
    "owner": "Engineering",
    "start_date": datetime(2023, 11, 1),
    "email": ["airflowresults@datacamp.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=20),
}

dag = DAG(
    "data_camp_course_schedule", default_args=default_args, schedule="30 12 * * 3"
)
