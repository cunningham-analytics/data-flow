from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

from scripts.data_extractor import run_data_extractor
from scripts.gsheets_loader import run_gsheets_loader

import logging

#Define default arguments
default_args = {
 'owner': 'ccunningham',
 'start_date': datetime (2024, 6, 30),
 'retries': 1,
}

data_start = '{{ data_interval_start }}'
data_end = '{{ data_interval_end }}'
dbt_project_path = os.environ['DBT_STOCKS_HOME']

# Instantiate your DAG
dag = DAG ('stock_data_pipeline',
           default_args=default_args,
           schedule="@daily"
           )

extract_stock_data = PythonOperator(
 task_id='extract_stock_data',
 python_callable=run_data_extractor,
 depends_on_past=True,
 wait_for_downstream=True,
 dag=dag,
)

run_data_transform = BashOperator(
 task_id="transform_pipeline",
 bash_command=F"""
 export DATA_INTERVAL_START='{data_start}'
 export DATA_INTERVAL_END='{data_end}'
 export DBT_PROJECT_PATH=
 dbt clean --project-dir {dbt_project_path}
 dbt deps {dbt_project_path}
 dbt build -s +int_gme__closing_calculations {dbt_project_path}
 """
)

load_to_gsheets = PythonOperator(
 task_id='load_to_gsheets',
 python_callable=run_gsheets_loader,
 dag=dag,
)

# Set task dependencies
extract_stock_data >> run_data_transform >> load_to_gsheets