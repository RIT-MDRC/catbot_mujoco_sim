# Catbot MuJoCo

An interactive MuJoCo workspace for building and previewing a small quadruped robot. The robot and its scene are authored as Jinja2 XML templates, then rendered in a reactive [Marimo](https://marimo.io) notebook.

## mjlab CPU rollout on macOS

For GPU PPO training on RIT Research Computing, use the new RSL-RL trainer:
see [RIT cluster training instructions](TRAINING_RIT.md). `uv run train` and
`uv run mjlab-train` run that trainer; `uv run train-sb3` retains the old trainer.

The checkpoint-free rollout uses **mjlab 1.6.0 / MuJoCo Warp** on the CPU,
with one simulation world. It reuses the existing Jinja MJCF templates unchanged:
the ball-jointed hips, all 16 actuator slots, locked hip-z targets, contacts,
and 0.002-second physics timestep are retained. The existing Gymnasium
environment, legacy PPO scripts, notebook, and manual viewer remain available.

```sh
uv sync --locked
uv run mjlab-rollout
# Move the front-left knee sinusoidally for 2 seconds of simulation:
uv run mjlab-rollout --motion sine --steps 1000
```

The default runs 500 physics steps (one simulated second) holding the initial
joint targets, then prints a JSON summary with device, elapsed simulation time,
base position, and finite-state status. These open-loop targets do not balance
the robot or produce a learned gait. CPU execution may be slower than real time;
the first run also compiles and caches Warp kernels.

To watch it in the native viewer on this Mac:

```sh
uv run mjlab-rollout --viewer --motion sine --steps 1000
```

The command automatically relaunches through `mjpython` on macOS and supplies
uv Python's shared-library directory to the launcher. The viewer closes when
the requested steps finish; closing it early stops the
rollout. Physics is advanced only by `mjlab.sim.Simulation.step()`; the native
MuJoCo data is a display copy. Viewer actuator sliders are not rollout inputs.
No checkpoint, CUDA device, or training run is needed. See mjlab's
[platform support notes](https://mujocolab.github.io/mjlab/v1.6.0/source/faq.html).

Run the real CPU integration check and existing environment tests with:

```sh
uv run python -m unittest discover -s tests
```

## Requirements

- [uv](https://docs.astral.sh/uv/) for Python and dependency management
- Python 3.14 or newer (uv installs a compatible Python automatically when needed)
- A graphical environment with OpenGL support for MuJoCo rendering

## Install

Clone the repository and enter it:

```sh
git clone <repository-url>
cd catbot_mujoco
```

Create the project environment and install the locked dependencies:

```sh
uv sync
```

`uv sync` creates `.venv/` and installs Marimo, MuJoCo, Jinja2, Pillow, and the development tooling declared in `pyproject.toml`.

## Run

Start the interactive notebook from the repository root:

```sh
uv run notebook
```

The `notebook` command launches `uv run marimo edit notebook/main.py --watch`. Marimo prints a local URL; open it in a browser to view the generated MuJoCo XML and rendered robot. Keep the command running while you work.

Editing either file in `assets/` automatically refreshes the dependent notebook cells:

```sh
# Robot body, joints, geometry, and appearance
assets/robot.xml.j2

# MuJoCo scene wrapper, lighting, and robot insertion point
assets/world.xml.j2
```

`main.py` is the root-level launcher used by the `notebook` command. Running `uv run python main.py` starts the same watched Marimo editor.

### Live MP4 preview

For a persistent browser video preview, open the dedicated livestream notebook:

```sh
uv run marimo edit notebook/livestream.py --watch
```

It starts a localhost-only backend, streams the simulation as fragmented H.264 MP4, and embeds the video in a Marimo iframe. The notebook provides Start / Stop and Reset controls. This mode requires a system `ffmpeg` installation with `libx264` support.


## Tensor Board

For training log, you can use tensor board:
```sh
uv run tensorboard --logdir runs
```

Training uses W&B by default for cloud dashboards and checkpoint uploads.
Use `--logger tensorboard` for local-only logging.
See [the W&B cluster setup](TRAINING_RIT.md#6-wb-dashboards-and-cloud-checkpoints)
for login, Slurm settings, offline sync, and downloading checkpoints. W&B mode
also keeps local TensorBoard logs.

## Workspace layout

```text
.
├── assets/
│   ├── robot.xml.j2   # Jinja template for the Catbot quadruped
│   └── world.xml.j2   # Jinja template for the enclosing MuJoCo scene
├── notebook/
│   ├── main.py        # Static reactive scene preview
│   └── livestream.py  # Local MP4 stream controls and iframe preview
├── livestream_backend.py # Local MuJoCo-to-MP4 streaming utility
├── main.py            # `uv run notebook` launcher
├── pyproject.toml     # Project metadata and Python dependencies
├── uv.lock            # Reproducible dependency lockfile
└── README.md
```

### How the simulation is assembled

1. `notebook/main.py` loads the templates from `assets/` with Jinja2.
2. It renders `robot.xml.j2` into a `<body>` definition.
3. It passes that body as `worldbody` while rendering `world.xml.j2` into a complete `<mujoco>` document.
4. MuJoCo parses the resulting XML, and the static notebook renders a camera view as an image.

`robot.xml.j2` defines a free-floating torso, four ball-jointed hips, hinge knees, capsule limbs, and spherical feet. `world.xml.j2` supplies the top light and the `<worldbody>` where the robot is placed.

## Legacy Stable-Baselines3 training

This section describes the old CPU trainer and its `.zip` checkpoints. For the
new GPU trainer and `.pt` checkpoints, follow [TRAINING_RIT.md](TRAINING_RIT.md).

`catbot_env.py` exposes the rendered model as `CatbotEnv`, a standard Gymnasium environment. It has 16 normalized position-action slots for checkpoint compatibility: hip x/y and knee targets are active, while every hip-z target is fixed at zero because that movement is unavailable on the real robot. Its 66-value observation contains base pose and velocity, joint pose and velocity, the current `(forward, lateral, yaw)` velocity command, and the prior action.

The initial task is velocity tracking. The reward combines velocity tracking, uprightness, a small alive bonus, and penalties for control magnitude and abrupt changes. Episodes terminate for a fall, inverted torso, or non-finite physics state, and truncate after 1,000 control steps by default. Reset randomizes the torso mass, ground friction, initial state, and command; pass `domain_randomization=False` for deterministic physics parameters.

Run the deterministic smoke tests and a random-action rollout:

```sh
uv run python -m unittest tests/test_catbot_env.py
```

Install dependencies and train a first policy:

```sh
uv sync
uv run train-sb3 --timesteps 1000000
```

The final model is written to `runs/catbot_ppo.zip`; periodic checkpoints are written to `runs/checkpoints/` every 25,000 timesteps. The `runs/` directory is intentionally Git-ignored but remains visible in Finder and the terminal. The training environment is headless; use `CatbotEnv(render_mode="rgb_array")` for evaluation frames or `render_mode="human"` from a desktop session for an interactive MuJoCo viewer.

Change the checkpoint interval, or disable intermediate saves with `--checkpoint-freq 0`:

```sh
uv run train-sb3 --checkpoint-freq 50000
```

Open the native MuJoCo viewer and play one episode from the default checkpoint:

```sh
uv run rollout
```

To render a selected checkpoint, pass its path as the only argument:

```sh
uv run rollout runs/checkpoints/catbot_ppo_<run-id>_25000_steps.zip
```

The viewer runs deterministic PPO actions with fixed physics. It holds the final pose after a fall or the 1,000-step limit so it can be inspected; close the viewer to return to the terminal.

### Manual limb flex test

Open a native viewer with direct control over each hip and knee actuator:

```sh
uv run ragdoll
```

In the MuJoCo viewer, open the Actuator/Control panel and drag the actuator sliders. The default script leaves `data.ctrl` under the viewer's control, so slider changes are applied directly to the simulation.

The manual viewer uses half-strength position servos with the existing joint damping for softer landings. Startup and reset targets match the initial joint pose to avoid kicking the knees during the drop. Tune compliance with `uv run ragdoll --stiffness-scale 0.3` (softer) or `--stiffness-scale 1` (original stiffness). This setting applies only to the manual viewer; training and policy rollout physics are unchanged. Manual targets do not actively balance the robot, so extreme poses can still tip it over.

For terminal-based control instead, use:

```sh
uv run ragdoll --terminal
```

The terminal then accepts commands while the viewer is open. Select a leg and axis, then nudge or set its position target:

```text
leg fl
axis knee
set -1.8
status
```

Use `axis x` or `axis y` to test the available hip actuators. Hip-z is locked at zero in both the policy model and viewer. `+` and `-` move the selected target by `0.1`; targets are clipped to their actuator ranges. `reset` restores the reference pose, and `quit` closes the command loop. The viewer's actuator/joint panels remain available for inspecting the resulting pose.

To continue a finished or interrupted run, pass its checkpoint and the number of *additional* timesteps to collect. The default output path overwrites that checkpoint only after the extra training is complete.

```sh
uv run train-sb3 --resume runs/catbot_ppo.zip --timesteps 1000000
```

PPO defaults to CPU because MuJoCo physics and this small MLP policy are CPU-heavy. On Apple Silicon, opt into Metal Performance Shaders after confirming availability with `uv run python -c 'import torch; print(torch.backends.mps.is_available())'`:

```sh
uv run train-sb3 --device mps
```

## Development commands

Run static checks with Ruff:

```sh
uv run ruff check .
```

After changing dependencies, update the lockfile and sync the environment:

```sh
uv lock
uv sync
```

Commit `uv.lock` with dependency changes so other contributors get the same resolved environment.

## Troubleshooting

- **MuJoCo cannot create a renderer:** Run from a desktop session or configure an appropriate off-screen OpenGL backend for your platform. The notebook needs a renderer even though it displays the result in the browser.
- **Template edits do not refresh:** Start Marimo from the repository root using the command above. The notebook watches `assets/robot.xml.j2` and `assets/world.xml.j2` relative to that directory.
- **Python version error:** Let uv manage Python (`uv python install 3.14`) and rerun `uv sync`.
