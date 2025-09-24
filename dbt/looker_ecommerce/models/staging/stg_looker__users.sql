SELECT *
FROM {{ source('looker_ecommerce', 'users') }}