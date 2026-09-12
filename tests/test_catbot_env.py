import unittest

import numpy as np

from catbot_env import CatbotEnv


class CatbotEnvTests(unittest.TestCase):
    def test_reset_and_random_rollout_are_finite(self) -> None:
        environment = CatbotEnv(max_episode_steps=100)
        observation, info = environment.reset(
            seed=7, options={"command": [0.2, 0.0, 0.0]}
        )
        self.assertTrue(environment.observation_space.contains(observation))
        self.assertEqual(info["command"].shape, (3,))

        for _ in range(100):
            observation, reward, terminated, truncated, _ = environment.step(
                environment.action_space.sample()
            )
            self.assertTrue(environment.observation_space.contains(observation))
            self.assertTrue(np.isfinite(reward))
            if terminated or truncated:
                observation, _ = environment.reset()
        environment.close()

    def test_seeded_resets_match(self) -> None:
        first = CatbotEnv(domain_randomization=True)
        second = CatbotEnv(domain_randomization=True)
        first_observation, first_info = first.reset(seed=42)
        second_observation, second_info = second.reset(seed=42)
        np.testing.assert_allclose(first_observation, second_observation)
        np.testing.assert_allclose(first_info["command"], second_info["command"])
        first.close()
        second.close()
