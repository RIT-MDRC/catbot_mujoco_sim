"""Local MuJoCo-to-MP4 livestream backend for the Marimo preview notebook."""

from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import mujoco
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route
import uvicorn


FRAME_RATE = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


@dataclass(frozen=True)
class Frame:
    """A rendered RGB frame and the simulation time it represents."""

    number: int
    pixels: bytes
    simulation_time: float


class LiveSimulation:
    """Owns MuJoCo state and produces one latest frame for all video clients."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._world_xml: str | None = None
        self._world_revision = 0
        self._running = False
        self._stopped = False
        self._frame: Frame | None = None
        self._frame_number = 0
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_world(self, world_xml: str) -> None:
        """Replace the scene; the renderer thread resets it to time zero."""
        with self._condition:
            self._world_xml = world_xml
            self._world_revision += 1
            self._last_error = None
            self._condition.notify_all()

    def set_running(self, running: bool) -> None:
        with self._condition:
            self._running = running
            self._condition.notify_all()

    def snapshot(self) -> dict[str, float | bool | str | None]:
        with self._condition:
            return {
                "running": self._running,
                "time": self._frame.simulation_time if self._frame else 0.0,
                "error": self._last_error,
            }

    def frames(self, after: int = -1) -> Iterator[Frame]:
        """Yield the newest rendered frame whenever it changes."""
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopped
                    or (self._frame is not None and self._frame.number > after)
                )
                if self._stopped:
                    return
                assert self._frame is not None
                frame = self._frame
            after = frame.number
            yield frame

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=2)

    def _publish(self, pixels: bytes, simulation_time: float) -> None:
        with self._condition:
            self._frame_number += 1
            self._frame = Frame(self._frame_number, pixels, simulation_time)
            self._condition.notify_all()

    def _run(self) -> None:
        renderer: mujoco.Renderer | None = None
        model: mujoco.MjModel | None = None
        data: mujoco.MjData | None = None
        current_world_revision = -1

        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopped
                        or self._world_revision != current_world_revision
                        or (self._running and current_world_revision >= 0)
                    )
                    if self._stopped:
                        return
                    world_xml = self._world_xml
                    world_revision = self._world_revision
                    running = self._running

                if world_revision != current_world_revision:
                    if renderer is not None:
                        renderer.close()
                    try:
                        model = mujoco.MjModel.from_xml_string(world_xml or "")
                        data = mujoco.MjData(model)
                        mujoco.mj_forward(model, data)
                        renderer = mujoco.Renderer(
                            model, height=FRAME_HEIGHT, width=FRAME_WIDTH
                        )
                        current_world_revision = world_revision
                    except Exception as error:  # surfaced through /health
                        with self._condition:
                            self._last_error = str(error)
                            self._running = False
                        current_world_revision = world_revision
                        continue

                if model is None or data is None or renderer is None:
                    continue

                if running:
                    mujoco.mj_step(
                        model,
                        data,
                        nstep=max(1, round((1 / FRAME_RATE) / model.opt.timestep)),
                    )

                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = [0.0, 0.0, 0.2]
                camera.distance = 1.2
                camera.elevation = -20
                camera.azimuth = 90
                renderer.update_scene(data, camera=camera)
                self._publish(renderer.render().tobytes(), data.time)

                with self._condition:
                    if not self._running:
                        self._condition.wait()
                if running:
                    time.sleep(1 / FRAME_RATE)
        finally:
            if renderer is not None:
                renderer.close()


class LivestreamBackend:
    """A local HTTP server that converts the latest simulation frames to MP4."""

    def __init__(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg is required for the MP4 livestream. Install ffmpeg and retry."
            )

        self.simulation = LiveSimulation()
        self._server: uvicorn.Server | None = None
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None

    @property
    def url(self) -> str:
        assert self.port is not None
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            return

        app = Starlette(
            routes=[
                Route("/", self._player),
                Route("/health", self._health),
                Route("/stream.mp4", self._stream),
            ]
        )
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen()
        self.port = self._socket.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._socket]}, daemon=True
        )
        self._thread.start()

    def set_world(self, world_xml: str) -> None:
        self.simulation.set_world(world_xml)

    def set_running(self, running: bool) -> None:
        self.simulation.set_running(running)

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        self.simulation.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._socket is not None:
            self._socket.close()

    async def _health(self, request: object) -> JSONResponse:
        del request
        return JSONResponse(self.simulation.snapshot())

    async def _player(self, request: object) -> HTMLResponse:
        del request
        return HTMLResponse(
            """<!doctype html>
<html><head><style>
  html, body { margin: 0; background: #101418; color: #e8edf2; }
  video { display: block; width: 100%; height: 100%; object-fit: contain; }
</style></head><body>
  <video autoplay muted playsinline controls src="/stream.mp4"></video>
</body></html>"""
        )

    async def _stream(self, request: object) -> StreamingResponse:
        del request
        return StreamingResponse(
            self._encoded_frames(),
            media_type="video/mp4",
            headers={"Cache-Control": "no-store", "Connection": "keep-alive"},
        )

    def _encoded_frames(self) -> Iterator[bytes]:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-framerate",
            str(FRAME_RATE),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def write_frames() -> None:
            try:
                assert process.stdin is not None
                for frame in self.simulation.frames():
                    process.stdin.write(frame.pixels)
                    process.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

        writer = threading.Thread(target=write_frames, daemon=True)
        writer.start()
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read1(16_384):
                yield chunk
        finally:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()


_backend: LivestreamBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> LivestreamBackend:
    """Get the process-wide local backend, starting it on first use."""
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = LivestreamBackend()
            _backend.start()
    return _backend


def _shutdown_backend() -> None:
    if _backend is not None:
        _backend.close()


atexit.register(_shutdown_backend)
