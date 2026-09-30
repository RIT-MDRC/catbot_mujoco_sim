"""RSL-RL logging to W&B with local TensorBoard and checkpoint files."""

from pathlib import Path

import wandb
from rsl_rl.utils.log_writer import LogWriter
from torch.utils.tensorboard import SummaryWriter


class CatbotWandbWriter(SummaryWriter, LogWriter):
    def __init__(self, log_dir: str, project: str, entity: str | None = None):
        super().__init__(log_dir, flush_secs=10)
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=Path(log_dir).name,
            dir=str(Path(log_dir).resolve()),
        )
        self.stopped = False
        self.run.define_metric("iteration")
        self.run.define_metric("*", step_metric="iteration")

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        super().add_scalar(tag, scalar_value, global_step, **kwargs)
        # RSL-RL also emits rewards with elapsed seconds as the step. Keep those
        # in TensorBoard; W&B plots use PPO iteration consistently.
        if not tag.endswith("/time"):
            self.run.log({"iteration": global_step, tag: float(scalar_value)})

    def store_config(self, env_cfg, train_cfg):
        self.run.config.update({"environment": env_cfg, "runner": train_cfg})
        for name in ("config.json", "model.xml"):
            self.save_file(str(Path(self.log_dir) / name))

    def save_file(self, path: str):
        file = Path(path).resolve()
        self.run.save(str(file), base_path=str(file.parent), policy="now")

    def save_model(self, model_path: str, it: int):
        # The runner calls this only after torch.save has completed.
        self.save_file(model_path)

    def stop(self, exit_code=0):
        if not self.stopped:
            self.stopped = True
            self.close()
            self.run.finish(exit_code=exit_code)
