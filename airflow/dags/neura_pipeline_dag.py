from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.models import Variable
from airflow.operators.docker_operator import DockerOperator

# Images (built earlier)
IMAGE_BASE  = Variable.get("NEURA_IMAGE_BASE",  default_var="neura:base")
IMAGE_MEDIA = Variable.get("NEURA_IMAGE_MEDIA", default_var="neura:media")

# Host paths for DockerOperator volume binds (must be absolute!)
PROJECT_HOST_DIR = os.environ.get("PROJECT_HOST_DIR", "/ABS/PATH/TO/YOUR/REPO")
DATA_HOST_DIR    = os.environ.get("DATA_HOST_DIR",    "/ABS/PATH/TO/YOUR/REPO/robot_data")

# Container paths (consistent with your CLI)
PROJECT_IN_CT = "/app"
DATA_IN_CT    = "/data"

# Output/artifacts inside the project tree
WORK_ROOT   = f"{PROJECT_IN_CT}/output"
DISC_MAN    = f"{WORK_ROOT}/manifest/episodes.parquet"
VALID_OUT   = f"{WORK_ROOT}/validation"
STATS_OUT   = f"{WORK_ROOT}/stats/global_stats.json"
NORM_OUT    = f"{WORK_ROOT}/normalized"
DATASET_OUT = f"{WORK_ROOT}/dataset"

default_args = {
    "owner": "neura",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="neura_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # trigger manually; change to "@daily" if you like
    catchup=False,
    max_active_runs=1,
    tags=["neura", "data-pipeline"],
    dagrun_timeout=timedelta(hours=4),
) as dag:

    docker_binds = [
        f"{PROJECT_HOST_DIR}:{PROJECT_IN_CT}",
        f"{DATA_HOST_DIR}:{DATA_IN_CT}",
    ]

    discover = DockerOperator(
        task_id="discover",
        image=IMAGE_BASE,
        api_version="auto",
        auto_remove=True,
        command=[
            "discover",
            "--data-root", DATA_IN_CT,
            "--manifest", DISC_MAN,
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        volumes=docker_binds,
        environment={"PYTHONUNBUFFERED": "1"},
    )

    validate = DockerOperator(
        task_id="validate",
        image=IMAGE_MEDIA,  # ffprobe-enabled
        api_version="auto",
        auto_remove=True,
        command=[
            "validate",
            "--manifest", DISC_MAN,
            "--meta-dir", f"{DATA_IN_CT}/meta",
            "--out", VALID_OUT,
            "--skip-video",  # remove this flag to enable ffprobe checks
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        volumes=docker_binds,
        environment={"PYTHONUNBUFFERED": "1"},
    )

    stats = DockerOperator(
        task_id="stats",
        image=IMAGE_BASE,
        api_version="auto",
        auto_remove=True,
        command=[
            "stats",
            "--data-root", DATA_IN_CT,
            "--validated-ids", f"{VALID_OUT}/validated_episodes.jsonl",
            "--out", STATS_OUT,
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        volumes=docker_binds,
        environment={"PYTHONUNBUFFERED": "1"},
    )

    align_transform = DockerOperator(
        task_id="align_transform",
        image=IMAGE_BASE,
        api_version="auto",
        auto_remove=True,
        command=[
            "align-transform",
            "--data-root", DATA_IN_CT,
            "--out", NORM_OUT,
            "--stats", STATS_OUT,
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        volumes=docker_binds,
        environment={"PYTHONUNBUFFERED": "1"},
    )

    materialize = DockerOperator(
        task_id="materialize",
        image=IMAGE_BASE,
        api_version="auto",
        auto_remove=True,
        command=[
            "materialize",
            "--norm-dir", NORM_OUT,
            "--out", DATASET_OUT,
            "--videos-root", f"{PROJECT_IN_CT}/robot_data/videos",   # repo-relative symlinks
            "--link-videos", "symlink",
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        volumes=docker_binds,
        environment={"PYTHONUNBUFFERED": "1"},
    )

    discover >> validate >> stats >> align_transform >> materialize
