select * from {{ ref('int_aapl__closing_calculations') }} union all  
select * from {{ ref('int_amzn__closing_calculations') }} union all  
select * from {{ ref('int_ffie__closing_calculations') }} union all  
select * from {{ ref('int_gme__closing_calculations') }} union all  
select * from {{ ref('int_googl__closing_calculations') }} union all  
select * from {{ ref('int_meta__closing_calculations') }} union all  
select * from {{ ref('int_msft__closing_calculations') }}
