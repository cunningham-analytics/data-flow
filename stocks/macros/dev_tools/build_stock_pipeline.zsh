#!/bin/zsh

ticker=$1

rm -f ${DBT_STOCKS_HOME}/models/staging/stg_${ticker}.sql
touch ${DBT_STOCKS_HOME}/models/staging/stg_${ticker}.sql
echo "{{ generate_staging_model(ticker='${ticker}')  }}" >> ${DBT_STOCKS_HOME}/models/staging/stg_${ticker}.sql


rm -f ${DBT_STOCKS_HOME}/models/intermediate/int_${ticker}__closing_calculations.sql
touch ${DBT_STOCKS_HOME}/models/intermediate/int_${ticker}__closing_calculations.sql
echo "{{ generate_closing_calculations_model(ticker='${ticker}')  }}" >> ${DBT_STOCKS_HOME}/models/intermediate/int_${ticker}__closing_calculations.sql


rm -f ${DBT_STOCKS_HOME}/models/intermediate/int_stocks__closing_calculations_final.sql
touch ${DBT_STOCKS_HOME}/models/intermediate/int_stocks__closing_calculations_final.sql
for file in ${DBT_STOCKS_HOME}/models/intermediate/int_*__closing_calculations.sql; echo "select * from {{ ref('${file:t:r}') }}" >> ${DBT_STOCKS_HOME}/models/intermediate/int_stocks__closing_calculations_final.sql
sed -i '' -e "$ ! s/$/ union all \\ /" ${DBT_STOCKS_HOME}/models/intermediate/int_stocks__closing_calculations_final.sql
