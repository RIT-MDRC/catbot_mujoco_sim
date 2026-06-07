import marimo

__generated_with = "0.23.6"
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
    asset_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
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

    def render_world(world_xml):
        """Renders the world in mujoco and returns the image for mo"""
        model = mujoco.MjModel.from_xml_string(world_xml)
        data = mujoco.MjData(model)

        with mujoco.Renderer(model) as renderer:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            pixels = renderer.render()
            img = Image.fromarray(pixels)

            return mo.image(img)

    return asset_file_watcher, print_xml, render_world


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
def _(render_world, world):
    render_world(world)
    return


if __name__ == "__main__":
    app.run()
