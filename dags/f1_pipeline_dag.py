from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'f1_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
}

with DAG(
    'f1_polesight_pipeline',
    default_args=default_args,
    description='Pipeline Otomatis F1: Mongo -> Spark -> Postgres (Full Data)',
    schedule='@daily', 
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['f1', 'spark', 'etl'],
) as dag:

    start_task = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Memulai Pipeline F1 PoleSight."'
    )

    # Task untuk menjalankan script Spark yang tadi kita buat
    run_spark_etl = BashOperator(
        task_id='run_spark_etl_job',
        bash_command='python /opt/airflow/jobs/etl_process.py'
    )

    end_task = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "Pipeline Selesai! Data Full Tersimpan."'
    )

    start_task >> run_spark_etl >> end_task