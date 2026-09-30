"""Check cloud upload wiring without network access or credentials."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training_wandb import CatbotWandbWriter


class WandbWriterTests(unittest.TestCase):
    def test_metadata_checkpoints_metrics_and_local_tensorboard(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("training_wandb.wandb.init") as init,
        ):
            root = Path(directory)
            for name in ("config.json", "model.xml", "model_25.pt"):
                (root / name).write_text("test data")
            writer = CatbotWandbWriter(directory, "catbot-test", "test-team")
            writer.store_config({"num_envs": 2}, {"seed": 7})
            writer.save_model(str(root / "model_25.pt"), 25)
            writer.add_scalar("Loss/value", 0.25, 25)
            writer.add_scalar("Train/mean_reward/time", 1.0, 100)
            writer.stop()
            writer.stop()

            run = init.return_value
            run.config.update.assert_called_once_with(
                {"environment": {"num_envs": 2}, "runner": {"seed": 7}}
            )
            self.assertEqual(
                {Path(call.args[0]).name for call in run.save.call_args_list},
                {"config.json", "model.xml", "model_25.pt"},
            )
            for call in run.save.call_args_list:
                self.assertEqual(call.kwargs["policy"], "now")
                self.assertEqual(call.kwargs["base_path"], str(root.resolve()))
            run.log.assert_called_once_with({"iteration": 25, "Loss/value": 0.25})
            run.finish.assert_called_once_with(exit_code=0)
            events = EventAccumulator(directory).Reload()
            self.assertEqual(events.Scalars("Loss/value")[0].step, 25)
            self.assertEqual(events.Scalars("Train/mean_reward/time")[0].step, 100)

    def test_failed_run_is_finished_with_error(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("training_wandb.wandb.init") as init,
        ):
            writer = CatbotWandbWriter(directory, "catbot-test")
            writer.stop(exit_code=1)
            init.return_value.finish.assert_called_once_with(exit_code=1)
