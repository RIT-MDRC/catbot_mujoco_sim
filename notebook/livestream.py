#! uv run --script --directory ../

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os

    import marimo as mo
    from jinja2 import Environment, FileSystemLoader

    from livestream_backend import get_backend

    asset_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    env = Environment(loader=FileSystemLoader(asset_folder))
    return env, get_backend, mo


@app.cell
def _(mo):
    def watch_asset(name: str):
        watch = mo.watch.file(f"assets/{name}")
        watch._file_name = name
        return watch

    return (watch_asset,)


@app.cell
def _(watch_asset):
    robot_watch = watch_asset("robot.xml.j2")
    world_watch = watch_asset("world.xml.j2")
    return robot_watch, world_watch


@app.cell
def _(env, robot_watch, world_watch):
    robot = env.get_template(robot_watch._file_name).render()
    world = env.get_template(world_watch._file_name).render(worldbody=robot)
    return (world,)


@app.cell(hide_code=True)
def _(mo):
    start_stop = mo.ui.button(
        value=False,
        on_click=lambda running: not running,
        label="Start / Stop",
    )
    reset = mo.ui.button(
        value=0,
        on_click=lambda count: count + 1,
        kind="warn",
        label="Reset",
    )
    mo.hstack([start_stop, reset], justify="start", gap=1)
    return reset, start_stop


@app.cell
def _(get_backend):
    backend = get_backend()
    return (backend,)


@app.cell
def _(backend, reset, world):
    # Both a template edit and Reset replace the backend's state at time zero.
    _ = reset.value
    backend.set_world(world)


@app.cell
def _(backend, start_stop):
    backend.set_running(start_stop.value)


@app.cell(hide_code=True)
def _(backend, mo, start_stop):
    status = "Running" if start_stop.value else "Paused"
    mo.md(f"**Simulation:** {status}  \\n+**Local stream:** `{backend.url}/stream.mp4`")


@app.cell(hide_code=True)
def _(backend, mo):
    mo.iframe(
        f'''<video autoplay muted playsinline
                  src="{backend.url}/stream.mp4"
                  style="width: 100%; height: 100%; object-fit: contain; background: #101418">
             Your browser cannot play the local MP4 stream.
           </video>''',
        height="500px",
    )


if __name__ == "__main__":
    app.run()
