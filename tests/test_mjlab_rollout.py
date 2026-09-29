"""Exercise real mjlab CPU physics with the existing Catbot model."""

import unittest

import mujoco
import numpy as np

from catbot_env import CatbotEnv
from mjlab_rollout import make_simulation, sync_viewer_data


class MjlabRolloutTests(unittest.TestCase):
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
