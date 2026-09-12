#! uv run --script --directory ../

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import mujoco
    from PIL import Image

    return Image, mo, mujoco


@app.cell
def _():
    import os

    from jinja2 import Environment, FileSystemLoader

    # 1. Tell Jinja to look for templates in the asset folder in the root of the project
    asset_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    file_loader = FileSystemLoader(asset_folder)
    env = Environment(loader=file_loader)
    return (env,)


@app.cell
def _(Image, mo, mujoco):
    def print_xml(xml):
        """pretty print xml string with syntax highlights"""
        return mo.md(f"""
    ```xml
    {xml}
    ```
    """)

    def asset_file_watcher(asset_path: str):
        """Creates a watch on a file in asset folder that reactively updates the cell and automatically runs any referenced cell blocks."""
        complete_path = f"assets/{asset_path}"
        watch = mo.watch.file(complete_path)
        watch._file_name = asset_path
        return watch

    def render_frame(model, data, camera):
        """Render the current MuJoCo state as a Marimo image."""
        with mujoco.Renderer(model) as renderer:
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            img = Image.fromarray(pixels)

            return mo.image(img)

    return asset_file_watcher, print_xml, render_frame


@app.cell
def _(mujoco):
    def controllable_camera():
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.2]
        camera.distance = 1.2
        camera.elevation = -20
        camera.azimuth = 90
        return camera

    cam = controllable_camera()
    return (cam,)


@app.cell
def _(asset_file_watcher):
    # mujoco xml model for the robot
    bot_model_file_path = "robot.xml.j2"
    bot_file_watcher = asset_file_watcher(bot_model_file_path)
    return (bot_file_watcher,)


@app.cell
def _(asset_file_watcher):
    # mujoco xml world model for simulation
    world_model_file_path = "world.xml.j2"
    world_file_watcher = asset_file_watcher(world_model_file_path)
    return (world_file_watcher,)


@app.cell
def _(bot_file_watcher, env, print_xml):
    bot_model_template = env.get_template(bot_file_watcher._file_name)
    bot = bot_model_template.render()
    print_xml(bot)
    return (bot,)


@app.cell
def _(bot, env, print_xml, world_file_watcher):
    world_model_template = env.get_template(world_file_watcher._file_name)
    world = world_model_template.render(worldbody=bot)
    print_xml(world)
    return (world,)


@app.cell
def _(cam, mujoco, render_frame, world):
    model = mujoco.MjModel.from_xml_string(world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    render_frame(model, data, cam)


@app.cell(hide_code=True)
def _(mo, world):
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    viewer_xml_path = Path(tempfile.gettempdir()) / "catbot_mujoco_viewer.xml"
    viewer_state = {"process": None}

    def open_viewer(_value):
        viewer_process = viewer_state["process"]

        if viewer_process is not None and viewer_process.poll() is None:
            return viewer_process.pid

        viewer_xml_path.write_text(world, encoding="utf-8")
        viewer_state["process"] = subprocess.Popen(  # pyright: ignore[reportArgumentType]
            [sys.executable, "-m", "mujoco.viewer", f"--mjcf={viewer_xml_path}"]
        )
        return viewer_state["process"].pid

    open_viewer_button = mo.ui.button(
        label="Open interactive MuJoCo viewer",
        on_click=open_viewer,
    )
    mo.vstack(
        [
            mo.md("Use the viewer's **Joint** panel or press `J` to show joint axes."),
            open_viewer_button,
        ]
    )


if __name__ == "__main__":
    app.run()
