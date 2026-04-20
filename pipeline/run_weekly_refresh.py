import os
import sys
import subprocess
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "pipeline", "logs")
LOG_PATH = os.path.join(LOG_DIR, "pipeline_runs.csv")

STEPS = [
    {
        "name": "Clean raw events",
        "script": os.path.join(BASE_DIR, "etl", "clean_events.py"),
    },
    {
        "name": "Build B2B marts",
        "script": os.path.join(BASE_DIR, "etl", "build_b2b_marts.py"),
    },
    {
        "name": "Build 30-day feature table",
        "script": os.path.join(BASE_DIR, "etl", "build_feature_table_30d.py"),
    },
    {
        "name": "Generate production churn scores",
        "script": os.path.join(BASE_DIR, "ml", "production_score_30d.py"),
    },
    {
        "name": "Build ranked churn scores",
        "script": os.path.join(BASE_DIR, "ml", "production_score_percentiles.py"),
    },
    {
        "name": "Load marts into Postgres",
        "script": os.path.join(BASE_DIR, "db", "load_marts_to_postgres.py"),
    },
]

def ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        df = pd.DataFrame(columns=[
            "run_id",
            "run_timestamp",
            "step_name",
            "status",
            "python_executable"
        ])
        df.to_csv(LOG_PATH, index=False)

def append_log(run_id, run_timestamp, step_name, status):
    df = pd.read_csv(LOG_PATH)
    new_row = pd.DataFrame([{
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "step_name": step_name,
        "status": status,
        "python_executable": sys.executable
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

def run_step(step_name: str, script_path: str, run_id: str, run_timestamp: str):
    print("\n" + "=" * 70)
    print(f"STARTING: {step_name}")
    print(f"SCRIPT:   {script_path}")
    print(f"PYTHON:   {sys.executable}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        text=True
    )

    if result.returncode != 0:
        append_log(run_id, run_timestamp, step_name, "FAILED")
        raise RuntimeError(f"Step failed: {step_name}")

    append_log(run_id, run_timestamp, step_name, "SUCCESS")
    print(f"COMPLETED: {step_name}")

def main():
    ensure_log_file()

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("WEEKLY CHURN PIPELINE STARTED")
    print("Run timestamp:", run_timestamp)
    print("Python executable:", sys.executable)
    print("=" * 70)

    for step in STEPS:
        run_step(step["name"], step["script"], run_id, run_timestamp)

    print("\n" + "=" * 70)
    print("WEEKLY CHURN PIPELINE FINISHED SUCCESSFULLY")
    print("Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

if __name__ == "__main__":
    main()