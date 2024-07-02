select * from {{ ref('int_gme__closing_calculations') }}   union all
select * from {{ ref('int_aapl__closing_calculations') }}