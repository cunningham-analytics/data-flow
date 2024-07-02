{% macro generate_staging_model(ticker) %}

{% set src_tbl = 'raw_'ticker %}

select
    cast(date as date) as report_date,
    round(cast(open as numeric), 2) as open_price,
    round(cast(close as numeric), 2) as close_price,
    round(cast(high as numeric), 2) as high_price,
    round(cast(low as numeric), 2) as low_price,
    volume,
    ticker
from {{ source('raw_store', src_tbl) }}
where cast(date as date) between cast('{{ env_var('INTERVAL_START') }}' as date) and cast('{{ env_var('INTERVAL_END') }}' as date)

{% endmacro %}