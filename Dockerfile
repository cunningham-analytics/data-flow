FROM python:3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends

WORKDIR /Users/christophercunningham/airflow_workspace/data-flow/
COPY . .

RUN pip3 install -r requirements.txt

WORKDIR /Users/christophercunningham/airflow_workspace/data-flow/stocks/
EXPOSE 8080
CMD        dbt deps && \
           dbt build --profiles-dir profiles --project-dir ${DBT_STOCKS_HOME} && \
           dbt docs generate --profiles-dir profiles --project-dir ${DBT_STOCKS_HOME} && \
           dbt docs serve --profiles-dir profiles --project-dir ${DBT_STOCKS_HOME}