{% set traffic_source_values = ['Adwords', 'Email', 'Facebook', 'Organic', 'YouTube'] %}
{% set browser_values = ['Chrome', 'Firefox', 'Safari', 'IE', 'Other'] %}

SELECT 
    user_id,
    COUNT(DISTINCT session_id) AS num_web_sessions,
        
    {%- for value in traffic_source_values %}
    COUNT(DISTINCT CASE WHEN traffic_source = '{{ value }}' THEN user_id END) AS num_traffic_source_{{ value }},
    {%- endfor %}
      
    {%- for value in browser_values %}
    COUNT(DISTINCT CASE WHEN browser = '{{ value }}' THEN user_id END) AS num_browser_{{ value }}
    {%- if not loop.last -%}
        ,
    {%- endif -%}
    {%- endfor %}

FROM {{ ref('stg_looker__events') }}
GROUP BY 1