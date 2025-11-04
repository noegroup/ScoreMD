from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
from typing import Any
from hydra_zen import to_yaml
import wandb
import logging
from scoremd.utils import slurm

log = logging.getLogger(__name__)


class SlurmInitializer(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        job_id = slurm.get_slurm_job_id()
        if job_id is not None:
            log.info(f"SLURM_JOB_ID: {job_id}")


class WandbInitializer(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        log.debug(to_yaml(config))

        if "wandb" in config and config.wandb.enabled:
            log.info("Initializing wandb...")
            init_args = OmegaConf.to_container(config.wandb)
            del init_args["enabled"]
            wandb.init(**init_args, config=OmegaConf.to_container(config))
            log.debug("Wandb initialized.")
