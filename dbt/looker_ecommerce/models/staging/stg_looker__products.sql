SELECT *
FROM {{ source('looker_ecommerce', 'products') }}
