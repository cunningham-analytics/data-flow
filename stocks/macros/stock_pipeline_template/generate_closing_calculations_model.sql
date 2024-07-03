{% macro generate_closing_calculations_model(ticker) %}

{% set mdl = 'stg_'ticker %}
with

chunk as (

    select
        *,
        -- need to use the index of the record for trading days - can't use date diffs bc of non-trading days
        row_number() over (order by report_date) as date_index
    from {{ ref(model) }}
)

select
    curr.ticker,
    curr.report_date,
    curr.close_price,
    curr.close_price - lag1.close_price as close_price__dod_delta,
    avg(prev20.close_price) as close_price__avg_prev20days,
    avg(prev50.close_price) as close_price__avg_prev50days
from
    chunk curr
left join chunk lag1
    on lag1.date_index = curr.date_index - 1
left join chunk prev20
    on prev20.date_index between curr.date_index - 21 and curr.date_index - 1
left join chunk prev50
    on prev50.date_index between curr.date_index - 51 and curr.date_index - 1
group by 1, 2, 3, 4

{% endmacro %}