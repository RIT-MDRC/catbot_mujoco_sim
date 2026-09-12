"""Interactive per-limb actuator tester for the Catbot MuJoCo model.

The native MuJoCo viewer remains responsible for the visual simulation. By
default, its built-in actuator controls edit ``data.ctrl`` directly. An
optional terminal command loop is retained for scripted/manual experiments.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterable

import mujoco
import numpy as np

from catbot_env import CatbotEnv

LEGS = ("fl", "fr", "rl", "rr")
# Hip-z actuators remain in the model for checkpoint/action compatibility, but
# their XML control range is fixed to zero and they are hidden from the selector.
AXES = ("x", "y", "knee")
DEFAULT_STEP = 0.1


class RagdollController:
    """Thread-safe actuator targets shared by the viewer and command loop."""

    def __init__(self, environment: CatbotEnv, step: float) -> None:
        self.environment = environment
        self.model = environment.model
        self.data = environment.data
        self.step = step
        self.selected_leg = "fl"
        self.selected_axis = "knee"
        mujoco.mj_forward(self.model, self.data)
        self.targets = np.clip(
            self.data.actuator_length.copy(),
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._actuator_ids = {
            f"{leg}_{axis}": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip_{axis}"
            )
            for leg in LEGS
            for axis in ("x", "y", "z")
        }
        self._actuator_ids.update(
            {
                f"{leg}_knee": mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{leg}_knee_actuator",
                )
                for leg in LEGS
            }
        )

    def __call__(self, _model: mujoco.MjModel, data: mujoco.MjData) -> None:
        with self._lock:
            data.ctrl[:] = self.targets

    def stop(self) -> None:
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

    def _id(self, leg: str | None = None, axis: str | None = None) -> int:
        return self._actuator_ids[f"{leg or self.selected_leg}_{axis or self.selected_axis}"]

    def select(self, value: str) -> None:
        value = value.lower()
        if value in LEGS:
            self.selected_leg = value
        elif value in AXES:
            self.selected_axis = value
        else:
            raise ValueError(f"select expects one of {LEGS + AXES}")
        print(f"Selected {self.selected_leg}.{self.selected_axis}")

    def adjust(self, delta: float) -> None:
        actuator_id = self._id()
        low, high = self.model.actuator_ctrlrange[actuator_id]
        with self._lock:
            self.targets[actuator_id] = np.clip(
                self.targets[actuator_id] + delta, low, high
            )
            target = self.targets[actuator_id]
        print(f"{self.selected_leg}.{self.selected_axis} target = {target:.3f}")

    def set_target(self, value: float) -> None:
        actuator_id = self._id()
        low, high = self.model.actuator_ctrlrange[actuator_id]
        with self._lock:
            self.targets[actuator_id] = np.clip(value, low, high)
            target = self.targets[actuator_id]
        print(
            f"{self.selected_leg}.{self.selected_axis} target = {target:.3f} "
            f"(requested {value:.3f}, range {low:.3f}..{high:.3f})"
        )

    def reset(self) -> None:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.targets[:] = np.clip(
                self.data.actuator_length,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            self.data.ctrl[:] = self.targets
            mujoco.mj_forward(self.model, self.data)
        print("Reset pose and actuator targets to their model references.")

    def status(self) -> None:
        print(f"Selected: {self.selected_leg}.{self.selected_axis}")
        for leg in LEGS:
            knee_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_knee"
            )
            knee_qpos = self.data.qpos[self.model.jnt_qposadr[knee_id]]
            foot_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot"
            )
            hip_targets = [self.targets[self._id(leg, axis)] for axis in ("x", "y")]
            knee_target = self.targets[self._id(leg, "knee")]
            print(
                f"  {leg}: hip xy target=({hip_targets[0]:+.2f}, "
                f"{hip_targets[1]:+.2f}), hip z=LOCKED, "
                f"knee target={knee_target:+.2f} | "
                f"knee={np.degrees(knee_qpos):+.1f} deg foot="
                f"({self.data.geom_xpos[foot_id][0]:+.2f}, "
                f"{self.data.geom_xpos[foot_id][1]:+.2f}, "
                f"{self.data.geom_xpos[foot_id][2]:+.2f})"
            )

    def run_commands(self) -> None:
        print_help()
        while not self.stopped():
            try:
                line = input("ragdoll> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.stop()
                break
            if not line:
                continue
            try:
                self.handle(line.split())
            except ValueError as error:
                print(f"Error: {error}")

    def handle(self, words: Iterable[str]) -> None:
        words = list(words)
        command = words[0].lower()
        if command in {"quit", "exit"}:
            self.stop()
        elif command in {"help", "?"}:
            print_help()
        elif command in {"status", "s"}:
            self.status()
        elif command in {"leg", "axis", "select"} and len(words) == 2:
            self.select(words[1])
        elif command in {"+", "-"} and len(words) == 1:
            self.adjust((1 if command == "+" else -1) * self.step)
        elif command in {"inc", "dec"} and len(words) == 1:
            self.adjust((1 if command == "inc" else -1) * self.step)
        elif command == "set" and len(words) == 2:
            self.set_target(float(words[1]))
        elif command == "reset" and len(words) == 1:
            self.reset()
        else:
            raise ValueError("unknown command; type 'help'")


def print_help() -> None:
    print(
        "Commands: leg fl|fr|rl|rr, axis x|y|knee, +/- (nudge selected), "
        "set VALUE, status, reset, help, quit"
    )
    print("Example: leg fl  -> axis knee  -> set -1.8  -> status")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the Catbot viewer and manually test each limb actuator."
    )
    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_STEP,
        help="Target increment used by +/- (default: 0.1).",
    )
    parser.add_argument(
        "--stiffness-scale",
        type=float,
        default=0.5,
        help="Position-servo stiffness multiplier (default: 0.5; original: 1).",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Use the terminal command loop instead of the viewer actuator controls.",
    )
    args = parser.parse_args()
    if args.step <= 0:
        raise SystemExit("--step must be positive")
    if not np.isfinite(args.stiffness_scale) or args.stiffness_scale <= 0:
        raise SystemExit("--stiffness-scale must be finite and positive")

    from mujoco import viewer

    environment = CatbotEnv(
        domain_randomization=False,
        action_smoothing_enabled=False,
    )
    mujoco.mj_resetData(environment.model, environment.data)
    environment.model.actuator_gainprm[:, 0] *= args.stiffness_scale
    environment.model.actuator_biasprm[:, 1] *= args.stiffness_scale
    controller = RagdollController(environment, args.step)
    environment.data.ctrl[:] = controller.targets
    mujoco.mj_forward(environment.model, environment.data)
    previous_callback = mujoco.get_mjcb_control()
    if args.terminal:
        mujoco.set_mjcb_control(controller)
        print("Opening viewer with terminal actuator control.")
    else:
        print(
            "Opening viewer with native actuator controls. Open the Actuator/Control "
            "panel and drag the sliders; changes are applied directly to the model."
        )
    try:
        if args.terminal:
            command_thread = threading.Thread(
                target=controller.run_commands, daemon=True
            )
            command_thread.start()
        viewer.launch(environment.model, environment.data)
    finally:
        controller.stop()
        mujoco.set_mjcb_control(previous_callback)
        environment.close()


if __name__ == "__main__":
    main()
