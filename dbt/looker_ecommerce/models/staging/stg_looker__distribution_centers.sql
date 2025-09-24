-- Best practices suggest minimizing transformations in staging dbt models, which should be just one layer of translation away from raw data sources like dbt seed and source files.
SELECT *
FROM {{ref('looker__distribution_centers')}}