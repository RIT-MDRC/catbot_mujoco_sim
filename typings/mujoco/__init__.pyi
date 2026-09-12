"""Typing surface for MuJoCo's generated native bindings."""

from collections.abc import Callable
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int32]
type ByteArray = NDArray[np.uint8]


class _MjOption:
    timestep: float


class _Actuator:
    name: str


class _Contact:
    geom1: int
    geom2: int


class MjModel:
    nu: int
    nv: int
    nq: int
    qpos0: FloatArray
    body_mass: FloatArray
    geom_friction: FloatArray
    actuator_ctrlrange: FloatArray
    jnt_qposadr: IntArray
    opt: _MjOption

    @classmethod
    def from_xml_string(cls, xml: str) -> MjModel: ...

    def actuator(self, index: int) -> _Actuator: ...


class MjData:
    qpos: FloatArray
    qvel: FloatArray
    ctrl: FloatArray
    xmat: FloatArray
    geom_xpos: FloatArray
    contact: list[_Contact]
    ncon: int
    time: float

    def __init__(self, model: MjModel) -> None: ...


class MjvCamera:
    type: int
    lookat: FloatArray
    distance: float
    elevation: float
    azimuth: float


class Renderer:
    def __init__(
        self, model: MjModel, height: int = ..., width: int = ...
    ) -> None: ...

    def __enter__(self) -> Self: ...
    def __exit__(self, *args: object) -> None: ...
    def update_scene(self, data: MjData, camera: MjvCamera | None = ...) -> None: ...
    def render(self) -> ByteArray: ...
    def close(self) -> None: ...


class mjtObj:
    mjOBJ_ACTUATOR: ClassVar[int]
    mjOBJ_BODY: ClassVar[int]
    mjOBJ_GEOM: ClassVar[int]
    mjOBJ_JOINT: ClassVar[int]


class mjtCamera:
    mjCAMERA_FREE: ClassVar[int]


type ControlCallback = Callable[[MjModel, MjData], None]


def get_mjcb_control() -> ControlCallback | None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_integratePos(
    model: MjModel, qpos: FloatArray, qvel: FloatArray, dt: float
) -> None: ...
def mj_name2id(model: MjModel, objtype: int, name: str) -> int: ...
def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData, nstep: int = ...) -> None: ...
def set_mjcb_control(callback: ControlCallback | None) -> None: ...
