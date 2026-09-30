"""CPU integration tests for the same batched environment/PPO path used on CUDA."""

import math
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
from mjlab_env import DEFAULT_CONFIG, CatbotMjlabEnv, load_catbot_config
from mjlab_train import runner_config


class ConfigTests(unittest.TestCase):
    def test_yaml_custom_values_and_mapping_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text(
                DEFAULT_CONFIG.read_text().replace(
                    "tracking_weight: 1.0", "tracking_weight: 2.5"
                )
            )
            config = load_catbot_config(path)
            self.assertEqual(config["tracking_weight"], 2.5)
            copied = load_catbot_config(config)
            copied["tracking_weight"] = 0
            self.assertEqual(config["tracking_weight"], 2.5)

    def test_invalid_config_rejected(self):
        for key, value in (
            ("motion_budget", 0),
            ("stance_radius", 0),
            ("frame_skip", 0),
            ("command_interval", 0),
            ("tracking_weight", -1),
            ("air_time_limit", 0.1),
            ("bound_speed", 0.1),
            ("action_smoothing", 1),
            ("gait_transition_seconds", 0),
            ("foot_clearance_target", 0),
        ):
            with self.subTest(key=key):
                config = load_catbot_config()
                config[key] = value
                with self.assertRaises(ValueError):
                    load_catbot_config(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("tracking_weight: 1\n")
            with self.assertRaises(ValueError):
                load_catbot_config(path)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = CatbotMjlabEnv(2, "cpu", seed=7)

    def test_saved_config_used_with_runtime_overrides(self):
        config = load_catbot_config()
        config["tracking_weight"] = 2.5
        config["frame_skip"] = 5
        env = CatbotMjlabEnv(1, "cpu", seed=9, config=config)
        self.assertEqual(env.cfg["tracking_weight"], 2.5)
        self.assertEqual(env.frame_skip, 5)
        self.assertEqual(env.cfg["num_envs"], 1)
        self.assertEqual(env.cfg["device"], "cpu")
        self.assertEqual(config["device"], "cuda:0")

    def test_observations_actions_and_partial_reset(self):
        env = self.env
        self.assertEqual(env.get_observations()["actor"].shape, (2, 110))
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
        env.feet_air_time[1] = 10  # Holding a foot up forfeits touchdown credit.
        _, reward = env.gait_terms(action, contact)
        torch.testing.assert_close(reward, torch.zeros(2))
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
        contact = torch.ones(2, 4, dtype=torch.bool)
        contact[:, :2] = False
        with patch.object(env, "foot_contacts", return_value=contact):
            env.step(torch.ones(2, 16))
            torch.testing.assert_close(
                env.feet_air_time[:, :2], torch.full((2, 2), env.dt)
            )
            torch.testing.assert_close(
                env.previous_action_delta[:, env.active_actions],
                torch.full((2, 12), 0.2),
            )
            env.step(torch.ones(2, 16))
            torch.testing.assert_close(
                env.feet_air_time[:, :2], torch.full((2, 2), 2 * env.dt)
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
        contact = torch.ones(2, 4, dtype=torch.bool)
        contact[:, :2] = False
        # Hold physics still to isolate the actual step/history/reward ordering.
        with (
            patch.object(env.sim, "step"),
            patch.object(env, "foot_contacts", return_value=contact),
        ):
            for _ in range(10):
                _, _, _, extras = env.step(torch.zeros(2, 16))
                self.assertEqual(float(extras["log"]["/air_time_reward"]), 0)
            contact.fill_(True)
            _, _, _, extras = env.step(torch.zeros(2, 16))
            self.assertAlmostEqual(
                float(extras["log"]["/air_time_reward"]),
                0.1 * (0.20 - 0.12) / (0.35 - 0.12),
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

    def test_held_leg_is_penalized_each_step_without_movement_command(self):
        env = self.env
        env.command.zero_()
        contact = torch.ones(2, 4, dtype=torch.bool)
        contact[:, 0] = False
        env.feet_air_time[:, 0] = 0.70
        env.sim.data.geom_xpos[:, env.feet, 2] = 0.10
        with (
            patch.object(env.sim, "step"),
            patch.object(env, "foot_contacts", return_value=contact),
        ):
            for _ in range(3):
                _, _, _, extras = env.step(torch.zeros(2, 16))
                self.assertAlmostEqual(
                    float(extras["log"]["/prolonged_swing_penalty"]), 0.5
                )
                self.assertAlmostEqual(float(env.feet_air_time[0, 0]), 0.70)
            contact.fill_(True)
            _, _, _, extras = env.step(torch.zeros(2, 16))
            self.assertEqual(float(extras["log"]["/prolonged_swing_penalty"]), 0)
            self.assertTrue((env.feet_air_time == 0).all())

    def test_normal_swing_has_no_overdue_cost(self):
        torch.testing.assert_close(
            self.env.swing_overdue(torch.tensor([0.12, 0.35, 0.525, 0.70])),
            torch.tensor([0.0, 0.0, 0.5, 1.0]),
        )

    def test_high_held_leg_cannot_dilute_other_foot_clearance_cost(self):
        env = self.env
        contact = torch.ones(2, 4, dtype=torch.bool)
        contact[:, 0] = False
        env.sim.data.geom_xpos[:, env.feet, 2] = 0.02
        action = torch.zeros(2, 16)
        _, _, first = env.reward_terms(action, contact)
        contact[:, 1] = False
        env.sim.data.geom_xpos[:, env.feet[1], 2] = 0.10
        _, _, second = env.reward_terms(action, contact)
        self.assertAlmostEqual(float(first["/foot_clearance_penalty"]), 0.05)
        torch.testing.assert_close(
            first["/foot_clearance_penalty"], second["/foot_clearance_penalty"]
        )

    def test_support_penalty_scales_with_missing_feet(self):
        env = self.env
        action = torch.zeros(2, 16)
        for count, expected in ((0, 1.0), (1, 0.5), (2, 0.0), (3, 0.0), (4, 0.0)):
            contact = torch.zeros(2, 4, dtype=torch.bool)
            contact[:, :count] = True
            reward, _, metrics = env.reward_terms(action, contact)
            self.assertAlmostEqual(float(metrics["/support_deficit_penalty"]), expected)
            env.cfg["support_deficit_weight"] = 0
            without_penalty, _, _ = env.reward_terms(action, contact)
            torch.testing.assert_close(
                without_penalty - reward, torch.full((2,), expected)
            )
            env.cfg["support_deficit_weight"] = 0.5

    def test_zero_support_does_not_terminate_or_override_timeout(self):
        env = self.env
        contact = torch.zeros(2, 4, dtype=torch.bool)
        with (
            patch.object(env.sim, "step"),
            patch.object(env, "foot_contacts", return_value=contact),
        ):
            _, reward, done, extras = env.step(torch.zeros(2, 16))
            self.assertFalse(done.any())
            self.assertTrue(torch.isfinite(reward).all())
            self.assertEqual(float(extras["log"]["/support_deficit_penalty"]), 1)
            env.max_episode_length = 2
            _, _, done, extras = env.step(torch.zeros(2, 16))
            self.assertTrue(done.all())
            self.assertTrue(extras["time_outs"].all())

    def test_trot_and_bound_pairing_in_both_directions(self):
        env = self.env
        for speed, expected in (
            (0.2, [1, 0, 0, 1]),
            (0.8, [1, 1, 0, 0]),
            (-0.8, [1, 1, 0, 0]),
        ):
            env.command[:] = torch.tensor([speed, 0, 0])
            env.gait_blend.copy_(env.target_gait_blend())
            env.gait_phase.zero_()
            target = env.gait_contact_target()
            self.assertEqual((target[0] > 0.5).int().tolist(), expected)
            env.gait_phase.fill_(0.5)
            self.assertEqual(
                (env.gait_contact_target()[0] > 0.5).int().tolist(),
                [1 - x for x in expected],
            )

    def test_gait_transition_is_rate_limited_and_keeps_support(self):
        env = self.env
        env.command[:] = torch.tensor([0.8, 0, 0])
        env.gait_blend.zero_()
        env.advance_gait()
        torch.testing.assert_close(env.gait_blend, torch.full((2,), 0.04))
        for blend in torch.linspace(0, 1, 11):
            env.gait_blend.fill_(float(blend))
            for phase in torch.linspace(0, 1, 41):
                env.gait_phase.fill_(float(phase))
                self.assertTrue(
                    ((env.gait_contact_target() > 0.5).sum(dim=1) >= 2).all()
                )

    def test_standing_target_and_gait_reset(self):
        env = self.env
        env.command.zero_()
        before = env.gait_phase.clone()
        env.advance_gait()
        torch.testing.assert_close(env.gait_phase, before)
        torch.testing.assert_close(env.gait_contact_target(), torch.ones(2, 4))
        before_phase = env.gait_phase[1].clone()
        before_blend = env.gait_blend[1].clone()
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.gait_phase[1], before_phase)
        torch.testing.assert_close(env.gait_blend[1], before_blend)
        self.assertTrue(0 <= env.gait_phase[0] < 1)
        self.assertAlmostEqual(
            float(env.gait_blend[0]), float(env.target_gait_blend()[0])
        )

    def test_phase_reward_prefers_correct_pair(self):
        env = self.env
        env.command[:] = torch.tensor([0.2, 0, 0])
        env.gait_phase.zero_()
        env.gait_blend.zero_()
        contact = torch.tensor([[True, False, False, True]]).expand(2, -1)
        _, _, correct = env.reward_terms(torch.zeros(2, 16), contact)
        _, _, wrong = env.reward_terms(torch.zeros(2, 16), ~contact)
        self.assertGreater(
            float(correct["/gait_contact_reward"]),
            float(wrong["/gait_contact_reward"]) + 0.7,
        )

    def test_command_resampling_preserves_phase_continuity(self):
        env = self.env
        env.cfg["command_interval"] = 1
        env.command[:] = torch.tensor([0.2, 0, 0])
        env.gait_blend.zero_()
        env.gait_phase.fill_(0.1)
        with patch.object(env.sim, "step"):
            observations, _, done, _ = env.step(torch.zeros(2, 16))
        self.assertFalse(done.any())
        torch.testing.assert_close(env.gait_phase, torch.full((2,), 0.13))
        self.assertFalse(
            torch.equal(env.command, torch.tensor([[0.2, 0, 0]]).expand(2, -1))
        )
        self.assertTrue((env.command[:, 0].abs() <= 0.8).all())
        self.assertEqual(observations["actor"].shape[1], 110)

    def test_heading_tracking_is_invariant_to_world_yaw(self):
        env = self.env
        action = torch.zeros(2, 16)
        contact = torch.ones(2, 4, dtype=torch.bool)
        for yaw in (0, math.pi / 2, -math.pi / 2, math.pi):
            env.sim.data.qpos[:, 3:7] = torch.tensor(
                [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            )
            for forward in (0.4, -0.4):
                lateral = 0.1
                env.command[:] = torch.tensor([forward, lateral, 0.2])
                env.sim.data.qvel[:, :6] = torch.tensor(
                    [
                        math.cos(yaw) * forward - math.sin(yaw) * lateral,
                        math.sin(yaw) * forward + math.cos(yaw) * lateral,
                        0,
                        0,
                        0,
                        0.2,
                    ]
                )
                _, _, metrics = env.reward_terms(action, contact)
                self.assertAlmostEqual(float(metrics["/tracking"]), 1.0)
                torch.testing.assert_close(
                    env.get_observations()["actor"][:, 5:8],
                    torch.tensor([[forward, lateral, 0.0]]).expand(2, -1),
                    atol=1e-6,
                    rtol=0,
                )
        # Facing world -X while moving world +X is backward, not forward.
        env.command[:] = torch.tensor([0.4, 0, 0])
        env.sim.data.qvel[:, :6] = torch.tensor([0.4, 0, 0, 0, 0, 0])
        _, _, metrics = env.reward_terms(action, contact)
        self.assertLess(float(metrics["/tracking"]), 0.3)

    def test_tilt_penalties_and_vertical_motion_do_not_reward_forward(self):
        env = self.env
        contact = torch.ones(2, 4, dtype=torch.bool)
        env.command.zero_()
        env.sim.data.qvel.zero_()
        for axis in (1, 2):
            for degrees in (0, 10, 20):
                half_angle = math.radians(degrees) / 2
                quat = torch.zeros(4)
                quat[0] = math.cos(half_angle)
                quat[axis] = math.sin(half_angle)
                env.sim.data.qpos[:, 3:7] = quat
                _, _, metrics = env.reward_terms(torch.zeros(2, 16), contact)
                expected = (
                    0.5
                    * (1 - math.cos(2 * half_angle))
                    / (1 - math.cos(math.radians(10)))
                )
                self.assertAlmostEqual(
                    float(metrics["/tilt_penalty"]), expected, places=5
                )
        env.sim.data.qvel[:, 2] = 1
        torch.testing.assert_close(
            env.heading_linear_velocity()[:, :2], torch.zeros(2, 2)
        )
        env.sim.data.qvel[:, 3:6] = torch.tensor([1.0, 2.0, 3.0])
        _, _, metrics = env.reward_terms(torch.zeros(2, 16), contact)
        self.assertAlmostEqual(float(metrics["/tilt_rate_penalty"]), 0.25)
        env.sim.data.qvel[:, 3:5] = 0
        _, _, metrics = env.reward_terms(torch.zeros(2, 16), contact)
        self.assertEqual(float(metrics["/tilt_rate_penalty"]), 0)

    def test_stance_circle_penalizes_only_grounded_feet_outside_radius(self):
        env = self.env
        contact = torch.ones(2, 4, dtype=torch.bool)
        hips = env.sim.data.xpos[:, env.hips, :2].clone()
        action = torch.zeros(2, 16)
        for distance, expected in (
            (0.0, 0.0),
            (0.05, 0.0),
            (0.1, 0.0),
            (0.15, 0.125),
            (0.2, 0.5),
        ):
            env.sim.data.geom_xpos[:, env.feet, :2] = hips + torch.tensor([distance, 0])
            _, _, metrics = env.reward_terms(action, contact)
            self.assertAlmostEqual(
                float(metrics["/stance_boundary_penalty"]), expected, places=5
            )
        # Only one of four feet contributes; swing feet are ignored.
        contact[:, 1:] = False
        _, _, metrics = env.reward_terms(action, contact)
        self.assertAlmostEqual(
            float(metrics["/stance_boundary_penalty"]), 0.125, places=5
        )
        _, _, metrics = env.reward_terms(action, torch.zeros_like(contact))
        self.assertEqual(float(metrics["/stance_boundary_penalty"]), 0)

    def test_stance_circle_follows_hip_and_is_rotation_invariant(self):
        env = self.env
        contact = torch.ones(2, 4, dtype=torch.bool)
        for yaw in (0, math.pi / 2, math.pi):
            env.sim.data.qpos[:, 0:2] = torch.tensor([2.0, -3.0])
            env.sim.data.qpos[:, 3:7] = torch.tensor(
                [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            )
            env.sim.forward()
            hip_xy = env.sim.data.xpos[:, env.hips, :2].clone()
            offset = torch.tensor([0.2 * math.cos(yaw), 0.2 * math.sin(yaw)])
            env.sim.data.geom_xpos[:, env.feet, :2] = hip_xy + offset
            _, _, metrics = env.reward_terms(torch.zeros(2, 16), contact)
            self.assertAlmostEqual(
                float(metrics["/stance_boundary_penalty"]), 0.5, places=5
            )
        env.cfg["stance_radius"] = 0.25
        _, _, metrics = env.reward_terms(torch.zeros(2, 16), contact)
        self.assertEqual(float(metrics["/stance_boundary_penalty"]), 0)

    def test_motion_budget_counts_travel_not_speed_or_other_joint_use(self):
        env = self.env
        travel = torch.zeros(2, 12)
        travel[:, 2] = 0.4
        env.accumulate_motion(travel)
        self.assertTrue((env.motion_cost == 0).all())
        travel[:, 2] = 0.3
        env.accumulate_motion(travel)
        torch.testing.assert_close(env.motion_cost, torch.full((2,), 0.2**2 / 12))
        before = env.motion_cost.clone()
        travel.zero_()
        travel[:, 0] = 0.3
        env.accumulate_motion(travel)
        torch.testing.assert_close(env.motion_cost, before)
        # The same travel split across many samples has the same total cost.
        env.motion_travel.zero_()
        env.motion_cost.zero_()
        travel.zero_()
        travel[:, 2] = 0.1
        for _ in range(7):
            env.accumulate_motion(travel)
        torch.testing.assert_close(env.motion_cost, before)

    def test_motion_window_reset_and_partial_episode_reset(self):
        env = self.env
        env.motion_travel.fill_(0.7)
        env.motion_progress[:] = torch.tensor([0.99, 0.1])
        with patch.object(env.sim, "step"):
            env.step(torch.zeros(2, 16))
        self.assertTrue((env.motion_travel[0] == 0).all())
        self.assertEqual(float(env.motion_progress[0]), 0)
        self.assertTrue((env.motion_travel[1] >= 0.7).all())
        before = env.motion_travel[1].clone()
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.motion_travel[1], before)
        self.assertTrue((env.motion_travel[0] == 0).all())

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
