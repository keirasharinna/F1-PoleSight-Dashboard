from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from pymongo import MongoClient

# DEFAULT ARGUMENTS

default_args = {
    'owner': 'f1_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
}

# HELPER FUNCTIONS

def check_if_new_data_exists(**context):
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://admin:password@mongo:27017/?authSource=admin")
        collection = client["f1_datalake"]["telemetry_raw"]
        
        # Check data count before & after (using XCom from previous task)
        ti = context['ti']
        count_before = ti.xcom_pull(task_ids='count_data_before')
        
        # Count current data
        count_after = collection.count_documents({})
        
        client.close()
        
        # If data increased, trigger ETL
        if count_after > count_before:
            print(f" New data detected! Before: {count_before}, After: {count_after}")
            return 'trigger_etl_pipeline'
        else:
            print(f" No new data. Count unchanged: {count_after}")
            return 'skip_etl'
            
    except Exception as e:
        print(f" Error checking data: {e}")
        return 'skip_etl'

def count_mongodb_data(**context):
    try:
        client = MongoClient("mongodb://admin:password@mongo:27017/?authSource=admin")
        collection = client["f1_datalake"]["telemetry_raw"]
        count = collection.count_documents({})
        client.close()
        return count
    except Exception as e:
        print(f"Error counting data: {e}")
        return 0

# DAG DEFINITION

with DAG(
    dag_id='f1_live_watcher',
    default_args=default_args,
    description='Auto-detect new F1 races and update data pipeline',
    schedule_interval='0 9 * * 1',  # Every Monday at 09:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['f1', 'live', 'ingestion', 'automated'],
    max_active_runs=1,  # Prevent multiple concurrent runs
) as dag:

    # Task 1: Count data BEFORE ingestion
    count_before = PythonOperator(
        task_id='count_data_before',
        python_callable=count_mongodb_data,
        provide_context=True
    )

    # Task 2: Run live ingestion script
    ingest_new_data = BashOperator(
        task_id='check_and_ingest_new_races',
        bash_command='python /opt/airflow/jobs/ingest_live.py',
        execution_timeout=timedelta(hours=2),  # Max 2 hours for download
    )

    # Task 3: Check if new data was actually added
    check_new_data = BranchPythonOperator(
        task_id='check_if_new_data_exists',
        python_callable=check_if_new_data_exists,
        provide_context=True
    )

    # Task 4: Trigger main ETL pipeline (if new data exists)
    trigger_etl = TriggerDagRunOperator(
        task_id='trigger_etl_pipeline',
        trigger_dag_id='f1_polesight_pipeline',
        wait_for_completion=False,
        reset_dag_run=True,  # Reset target DAG state
        execution_date='{{ ds }}',  # Pass execution date
        conf={'triggered_by': 'live_watcher'}  # Pass metadata
    )

    # Task 5: Skip ETL (if no new data)
    skip_etl = DummyOperator(
        task_id='skip_etl'
    )

    # Task 6: End task (convergence point)
    end = DummyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success'  # Run if any upstream succeeds
    )

    # TASK DEPENDENCIES

    count_before >> ingest_new_data >> check_new_data
    check_new_data >> [trigger_etl, skip_etl]
    [trigger_etl, skip_etl] >> end

# DOCUMENTATION

dag.doc_md = """
# F1 PoleSight - Live Race Watcher

## Purpose
Automatically detect and ingest new F1 qualifying sessions.

## Schedule
- **When**: Every Monday at 09:00 AM (after race weekend)
- **What**: Check for completed races in current season
- **Action**: Download new data → Trigger ETL pipeline

## Task Flow
1. Count existing data in MongoDB
2. Run ingestion script (checks F1 schedule vs DB)
3. Check if new data was added
4. If yes → Trigger main ETL pipeline
5. If no → Skip (system already up-to-date)

## Triggers
- Automatically triggers: `f1_polesight_pipeline` (main ETL)

## Logs
- Location: `/opt/airflow/logs/live_ingestion_YYYYMMDD.log`
- Check logs if ingestion fails

## Manual Trigger
Can be triggered manually from Airflow UI if needed.
"""