"""CPU integration tests for the same batched environment/PPO path used on CUDA."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

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
        self.assertEqual(env.get_observations()["actor"].shape, (2, 66))
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
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.sim.data.qpos[1], before)
        self.assertEqual(env.episode_length_buf.tolist(), [0, 1])

    def test_timeouts_and_falls(self):
        env = self.env
        env.max_episode_length = 1
        _, _, done, extras = env.step(torch.zeros(2, 16))
        self.assertTrue(done.all())
        self.assertTrue(extras["time_outs"].all())
        self.assertEqual(env.episode_length_buf.tolist(), [0, 0])
        env.max_episode_length = 1000
        env.sim.data.qpos[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        obs, _, done, extras = env.step(torch.zeros(2, 16))
        self.assertTrue(done[0])
        self.assertFalse(extras["time_outs"][0])
        self.assertTrue(torch.isfinite(obs["actor"]).all())

    def test_reward_matches_existing_environment(self):
        env = self.env
        original = CatbotEnv(domain_randomization=False)
        try:
            # Evaluate at the same contact-rich pose with zero foot velocity.
            env.sim.data.qpos[:, 2] = 0.20
            env.sim.forward()
            env.previous_feet.copy_(env.sim.data.geom_xpos[:, env.feet])
            action = torch.zeros(2, 16)
            rewards, ups, _ = env.reward_terms(action)
            for index in range(2):
                original.data.qpos[:] = env.sim.data.qpos[index].numpy()
                original.data.qvel[:] = env.sim.data.qvel[index].numpy()
                mujoco.mj_forward(original.model, original.data)
                original._command = env.command[index].numpy().astype(np.float64)
                original._previous_foot_positions = original.data.geom_xpos[
                    original._foot_ids
                ].copy()
                expected, _ = original._compute_reward(
                    np.zeros(16), float(ups[index]), float(original.data.qpos[2])
                )
                self.assertAlmostEqual(float(rewards[index]), expected, places=5)
                self.assertEqual(
                    env.foot_contacts()[index].tolist(),
                    [
                        original._foot_contacts_ground(int(foot))
                        for foot in original._foot_ids
                    ],
                )
        finally:
            original.close()

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
