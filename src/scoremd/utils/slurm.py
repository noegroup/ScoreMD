import os
from typing import Optional


def get_slurm_job_id() -> Optional[str]:
    if "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"]
    return None
