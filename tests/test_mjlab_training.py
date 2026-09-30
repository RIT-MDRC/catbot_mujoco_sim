"""CPU integration tests for the same batched environment/PPO path used on CUDA."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from catbot_env import CatbotEnv
from mjlab_env import CatbotMjlabEnv
from mjlab_train import runner_config


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = CatbotMjlabEnv(2, "cpu", seed=7)

    def test_observations_actions_and_partial_reset(self):
        env = self.env
        self.assertEqual(env.get_observations()["actor"].shape, (2, 94))
        _, reward, done, _ = env.step(torch.ones(2, 16))
        self.assertTrue(torch.isfinite(reward).all())
        self.assertFalse(done.any())
        np.testing.assert_allclose(
            env.sim.data.ctrl.numpy(),
            (env.center + 0.2 * env.half_range).expand(2, -1).numpy(),
            atol=1e-6,
        )
        locked = env.half_range == 0
        self.assertTrue((env.sim.data.ctrl[:, locked] == 0).all())
        before = env.sim.data.qpos[1].clone()
        env.feet_air_time.fill_(0.2)
        env.previous_action_delta.fill_(0.3)
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.sim.data.qpos[1], before)
        self.assertEqual(env.episode_length_buf.tolist(), [0, 1])
        self.assertTrue((env.feet_air_time[0] == 0).all())
        self.assertTrue((env.previous_action_delta[0] == 0).all())
        torch.testing.assert_close(env.feet_air_time[1], torch.full((4,), 0.2))
        torch.testing.assert_close(env.previous_action_delta[1], torch.full((16,), 0.3))

    def test_timeouts_and_falls(self):
        env = self.env
        env.max_episode_length = 1
        _, _, done, extras = env.step(torch.zeros(2, 16))
        self.assertTrue(done.all())
        self.assertTrue(extras["time_outs"].all())
        self.assertEqual(env.episode_length_buf.tolist(), [0, 0])
        env.max_episode_length = 1000
        env.sim.data.qpos[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        obs, reward, done, extras = env.step(torch.zeros(2, 16))
        self.assertTrue(done[0])
        self.assertEqual(float(reward[0]), -1)
        self.assertFalse(extras["time_outs"][0])
        self.assertTrue(torch.isfinite(obs["actor"]).all())

    def test_contacts_match_mujoco(self):
        env = self.env
        original = CatbotEnv(domain_randomization=False)
        try:
            env.sim.data.qpos[:, 2] = 0.20
            env.sim.forward()
            for index in range(2):
                original.data.qpos[:] = env.sim.data.qpos[index].numpy()
                mujoco.mj_forward(original.model, original.data)
                self.assertEqual(
                    env.foot_contacts()[index].tolist(),
                    [
                        original._foot_contacts_ground(int(foot))
                        for foot in original._foot_ids
                    ],
                )
        finally:
            original.close()

    def test_air_time_is_bounded_touchdown_only_and_command_gated(self):
        env = self.env
        env.command[:] = torch.tensor([0.4, 0.0, 0.0])
        contact = torch.ones(2, 4, dtype=torch.bool)
        env.previous_contact.zero_()
        env.has_stance.fill_(True)
        env.feet_air_time[0] = 0.08  # Tiny shuffling steps earn nothing.
        env.feet_air_time[1] = 0.35
        action = torch.zeros(2, 16)
        _, reward = env.gait_terms(action, contact)
        torch.testing.assert_close(reward, torch.tensor([0.0, 1.0]))
        env.feet_air_time[1] = 10  # No unbounded reward for hopping/hovering.
        _, reward = env.gait_terms(action, contact)
        torch.testing.assert_close(reward, torch.tensor([0.0, 1.0]))
        _, reward = env.gait_terms(action, ~contact)
        self.assertTrue((reward == 0).all())
        env.feet_air_time.zero_()
        _, reward = env.gait_terms(action, contact)
        self.assertTrue((reward == 0).all())
        env.feet_air_time.fill_(0.35)
        env.command.zero_()
        _, reward = env.gait_terms(action, contact)
        self.assertTrue((reward == 0).all())

    def test_action_reversal_costs_more_than_constant_change(self):
        env = self.env
        env.previous_action.fill_(0.2)
        env.previous_action_delta.fill_(0.1)
        contact = torch.zeros(2, 4, dtype=torch.bool)
        steady, _ = env.gait_terms(torch.full((2, 16), 0.3), contact)
        reversal, _ = env.gait_terms(torch.full((2, 16), 0.1), contact)
        torch.testing.assert_close(steady, torch.zeros(2), atol=1e-7, rtol=0)
        torch.testing.assert_close(reversal, torch.full((2,), 0.04))

    def test_step_counts_air_samples_and_clears_swing_on_touchdown(self):
        env = self.env
        env.previous_contact.fill_(True)
        contact = torch.zeros(2, 4, dtype=torch.bool)
        with patch.object(env, "foot_contacts", return_value=contact):
            env.step(torch.ones(2, 16))
            torch.testing.assert_close(env.feet_air_time, torch.full((2, 4), env.dt))
            torch.testing.assert_close(
                env.previous_action_delta[:, env.active_actions],
                torch.full((2, 12), 0.2),
            )
            env.step(torch.ones(2, 16))
            torch.testing.assert_close(
                env.feet_air_time, torch.full((2, 4), 2 * env.dt)
            )
            torch.testing.assert_close(
                env.previous_action_delta[:, env.active_actions],
                torch.full((2, 12), 0.16),
            )
            contact.fill_(True)
            env.step(torch.ones(2, 16))
            self.assertTrue((env.feet_air_time == 0).all())

    def test_clearance_shortfall_has_meaningful_penalty(self):
        env = self.env
        env.sim.data.geom_xpos[:, env.feet, 2] = 0.02
        _, _, metrics = env.reward_terms(
            torch.zeros(2, 16), torch.zeros(2, 4, dtype=torch.bool)
        )
        self.assertAlmostEqual(float(metrics["/foot_clearance_penalty"]), 0.2)

    def test_initial_landing_is_not_rewarded(self):
        env = self.env
        env.has_stance.zero_()
        env.feet_air_time.fill_(0.35)
        env.command[:] = torch.tensor([0.4, 0.0, 0.0])
        _, reward = env.gait_terms(
            torch.zeros(2, 16), torch.ones(2, 4, dtype=torch.bool)
        )
        self.assertTrue((reward == 0).all())

    def test_touchdown_reward_is_paid_once_after_counted_swing(self):
        env = self.env
        env.command[:] = torch.tensor([0.4, 0.0, 0.0])
        env.has_stance.fill_(True)
        env.previous_contact.fill_(True)
        contact = torch.zeros(2, 4, dtype=torch.bool)
        # Hold physics still to isolate the actual step/history/reward ordering.
        with patch.object(env.sim, "step"), patch.object(
            env, "foot_contacts", return_value=contact
        ):
            for _ in range(10):
                _, _, _, extras = env.step(torch.zeros(2, 16))
                self.assertEqual(float(extras["log"]["/air_time_reward"]), 0)
            contact.fill_(True)
            _, _, _, extras = env.step(torch.zeros(2, 16))
            self.assertAlmostEqual(
                float(extras["log"]["/air_time_reward"]),
                0.2 * (0.20 - 0.12) / (0.35 - 0.12),
                places=6,
            )
            _, _, _, extras = env.step(torch.zeros(2, 16))
            self.assertEqual(float(extras["log"]["/air_time_reward"]), 0)

    def test_clearance_measures_bottom_and_masks_stance(self):
        env = self.env
        action = torch.zeros(2, 16)
        contact = torch.zeros(2, 4, dtype=torch.bool)
        env.sim.data.geom_xpos[:, env.feet, 2] = 0.08
        _, _, metrics = env.reward_terms(action, contact)
        self.assertAlmostEqual(float(metrics["/foot_clearance_penalty"]), 0)
        env.sim.data.geom_xpos[:, env.feet, 2] = 0.05
        _, _, metrics = env.reward_terms(action, contact)
        self.assertAlmostEqual(float(metrics["/foot_clearance_penalty"]), 0.05)
        _, _, metrics = env.reward_terms(action, ~contact)
        self.assertAlmostEqual(float(metrics["/foot_clearance_penalty"]), 0)

    def test_history_is_observed_and_locked_actions_are_ignored(self):
        env = self.env
        before = env.get_observations()["actor"].clone()
        env.previous_action_delta.fill_(0.1)
        env.feet_air_time.fill_(0.2)
        self.assertFalse(torch.equal(before, env.get_observations()["actor"]))
        env.previous_action_delta.zero_()
        action = torch.zeros(2, 16)
        action[:, ~env.active_actions] = 1
        acceleration, _ = env.gait_terms(action, env.previous_contact)
        self.assertTrue((acceleration == 0).all())
        env.step(action)
        self.assertTrue((env.previous_action == 0).all())

    def test_ppo_update_checkpoint_and_resume(self):
        config = runner_config(steps=4, save_interval=1, logger="tensorboard")
        with tempfile.TemporaryDirectory() as directory:
            runner = OnPolicyRunner(self.env, deepcopy(config), directory, device="cpu")
            before = [
                parameter.detach().clone()
                for parameter in runner.alg.actor.parameters()
            ]
            runner.learn(1)
            self.assertTrue(
                any(
                    not torch.equal(old, new)
                    for old, new in zip(before, runner.alg.actor.parameters())
                )
            )
            checkpoint = Path(directory) / "model_0.pt"
            self.assertTrue(checkpoint.is_file())
            restored = OnPolicyRunner(self.env, deepcopy(config), device="cpu")
            restored.load(str(checkpoint), map_location="cpu")
            for a, b in zip(
                runner.alg.actor.parameters(), restored.alg.actor.parameters()
            ):
                torch.testing.assert_close(a, b)
            self.assertTrue(restored.alg.optimizer.state)
            actions = restored.get_inference_policy()(self.env.get_observations())
            self.assertEqual(actions.shape, (2, 16))
            self.assertTrue(torch.isfinite(actions).all())


if __name__ == "__main__":
    unittest.main()
