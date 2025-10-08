# Introduction to Apache Airflow in Python

## Intro to Airflow

## Implementing Airflow DAGs

## Building Pipeline

* Jinja template
* Airflow built-in variables
  * Execution Date: {{ ds }} -> YYYY-MM-DD
  * Execution Date, no dashes: {{ ds_nodash }} -> YYYYMMDD
  * Prev Execution Date: {{ prev_ds }} -> YYYY-MM-DD
  * Prev Execution Date, no dashes: {{ prev_ds_nodash }} -> YYYYMMDD
  * DAG object: {{ dag }}
  * Airflow config object: {{ conf }}
* Macros
  * {{ macro }}
  * {{ macro.datetime }}
  * {{ macro.timedelta }}
  * {{ macro.uuid }}
  * {{ macro.ds_add('2020-04-15', 5)}}
    * modify days from a date, this example returns 2020-04-20
