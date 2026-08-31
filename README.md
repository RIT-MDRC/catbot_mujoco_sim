# Catbot MuJoCo

An interactive MuJoCo workspace for building and previewing a small quadruped robot. The robot and its scene are authored as Jinja2 XML templates, then rendered in a reactive [Marimo](https://marimo.io) notebook.

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
