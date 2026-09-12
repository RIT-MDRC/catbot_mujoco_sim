from . import MjData, MjModel

def launch(
    model: MjModel | None = ...,
    data: MjData | None = ...,
    *,
    loader: object = ...,
    show_left_ui: bool = ...,
    show_right_ui: bool = ...,
) -> None: ...


class Viewer:
    def sync(self) -> None: ...
    def close(self) -> None: ...


def launch_passive(model: MjModel, data: MjData) -> Viewer: ...
