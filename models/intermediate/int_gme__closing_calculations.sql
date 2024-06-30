

with

chunk as (

    select *
    from {{ ref('stg_gme') }}
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
    on lag1.report_date = curr.report_date - INTERVAL '1 day'
left join chunk prev20
    on prev20.report_date between curr.report_date - INTERVAL '21 day' and curr.report_date - INTERVAL '1 day'
left join chunk prev50
    on prev50.report_date between curr.report_date - INTERVAL '51 day' and curr.report_date - INTERVAL '1 day'
group by 1, 2, 3, 4