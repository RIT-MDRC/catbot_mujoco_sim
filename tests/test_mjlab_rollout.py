"""Exercise real mjlab CPU physics with the existing Catbot model."""

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from catbot_env import CatbotEnv
from mjlab_rollout import KeyboardCommand, make_simulation, sync_viewer_data


class MjlabRolloutTests(unittest.TestCase):
    def test_keyboard_commands_and_reset_override(self):
        control = KeyboardCommand()
        env = SimpleNamespace(command=torch.ones(1, 3))
        control.apply(env)
        torch.testing.assert_close(env.command, torch.zeros(1, 3))
        for key in "WAQ":
            control.on_key(ord(key))
        control.apply(env)
        torch.testing.assert_close(env.command, torch.tensor([[0.1, 0.05, 0.1]]))
        env.command.fill_(-0.7)  # An episode reset samples a random command.
        control.apply(env)
        torch.testing.assert_close(env.command, torch.tensor([[0.1, 0.05, 0.1]]))
        for _ in range(30):
            for key in "WDE":
                control.on_key(ord(key))
        control.apply(env)
        torch.testing.assert_close(env.command, torch.tensor([[0.8, -0.1, -0.8]]))
        for _ in range(30):
            control.on_key(ord("S"))
        control.apply(env)
        self.assertAlmostEqual(float(env.command[0, 0]), -0.8)
        control.on_key(ord(" "))
        control.apply(env)
        torch.testing.assert_close(env.command, torch.zeros(1, 3))

    def test_model_state_and_reset(self):
        simulation = make_simulation()
        original = mujoco.MjModel.from_xml_string(CatbotEnv._render_mjcf())
        self.assertEqual(simulation.device, "cpu")
        self.assertEqual(simulation.num_envs, 1)
        for field in ("jnt_type", "actuator_gear", "actuator_ctrlrange", "body_mass"):
            np.testing.assert_array_equal(
                getattr(original, field), getattr(simulation.mj_model, field)
            )
        initial = simulation.data.qpos[0].numpy().copy()
        for _ in range(20):
            simulation.step()
        self.assertAlmostEqual(float(simulation.data.time[0]), 0.04, places=6)
        self.assertTrue(np.isfinite(simulation.data.qpos.numpy()).all())
        self.assertFalse(np.array_equal(initial, simulation.data.qpos[0].numpy()))
        sync_viewer_data(simulation)
        np.testing.assert_array_equal(
            simulation.mj_data.qpos, simulation.data.qpos[0].numpy()
        )
        simulation.reset()
        np.testing.assert_allclose(
            simulation.data.qpos[0].numpy(), original.qpos0, atol=1e-7
        )
        self.assertEqual(float(simulation.data.time[0]), 0)


if __name__ == "__main__":
    unittest.main()
