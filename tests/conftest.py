from __future__ import annotations

import math

import numpy as np
import pytest

from viplanner_model.config import VIPlannerSemMetaHandler

_COLORS = VIPlannerSemMetaHandler().class_color
_SURFACES = ("floor", "wall", "ceiling", "wall")


def corridor(
    width: int, height: int, hfov_deg: float = 90.0, camera_height: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Depth in meters and the class-color image of a 4 m wide, 2.5 m high corridor ending 8 m ahead."""
    focal = width / 2 / math.tan(math.radians(hfov_deg) / 2)
    u = (np.arange(width, dtype=np.float32) - (width - 1) / 2) / focal
    v = (np.arange(height, dtype=np.float32) - (height - 1) / 2) / focal
    left, up = np.meshgrid(-u, -v)
    with np.errstate(divide="ignore"):
        hits = np.stack(
            [
                np.where(up < 0, -camera_height / up, np.inf),
                np.where(left < 0, -2.0 / left, np.inf),
                np.where(up > 0, (2.5 - camera_height) / up, np.inf),
                np.where(left > 0, 2.0 / left, np.inf),
                np.full_like(up, 8.0),
            ]
        )
    surface = hits.argmin(axis=0)
    depth = hits.min(axis=0).astype(np.float32)
    colors = np.array([_COLORS[name] for name in (*_SURFACES, "wall")], dtype=np.uint8)
    return depth, colors[surface]


@pytest.fixture(scope="session")
def corridor_640x480() -> tuple[np.ndarray, np.ndarray]:
    return corridor(640, 480)
