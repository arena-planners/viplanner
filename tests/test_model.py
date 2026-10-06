from __future__ import annotations

import contextlib
import pathlib
import time

import numpy as np
import pytest
import torch
import yaml
from arena_planners.sdk import Step
from conftest import corridor
from PIL import Image

import planner
from viplanner_model.config import VIPlannerSemMetaHandler

_DIR = pathlib.Path(planner.__file__).parent
_DATA = pathlib.Path(__file__).parent / "data"
_COLORS = VIPlannerSemMetaHandler().class_color


def _missing_weights() -> list[str]:
    with open(_DIR / "weights.yaml") as fh:
        dests = [entry["dest"] for entry in yaml.safe_load(fh)["files"]]
    return [dest for dest in dests if not (_DIR / dest).is_file()]


@pytest.fixture(scope="module")
def viplanner() -> planner.ViPlanner:
    missing = _missing_weights()
    if missing:
        pytest.skip(f"weights {missing[:2]}... missing, run `arena feature planners add viplanner`")
    if not torch.cuda.is_available():
        pytest.skip("the segmenter needs a CUDA device")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    planner.on_init({"device": "cuda", "camera_offset_x": 0.03, "camera_offset_y": 0.0, "fear_threshold": 0.7})
    allocated = torch.cuda.memory_allocated() / 2**30
    print(f"\nviplanner load {time.perf_counter() - start:.1f} s, {allocated:.2f} GiB allocated")
    assert planner._planner is not None
    return planner._planner


def _street() -> np.ndarray:
    return np.asarray(Image.open(_DATA / "street.jpg").convert("RGB"))


def _street_features(depth: np.ndarray, goal_y: float = 0.0) -> dict:
    rgb = np.asarray(Image.open(_DATA / "street.jpg").convert("RGB").resize((depth.shape[1], depth.shape[0])))
    planner.on_reset("street", {"goal_pose": {"x": 4.0, "y": goal_y, "theta": 0.0}})
    return {"robot_pose": np.zeros(3), "depth": depth, "rgb": rgb}


@contextlib.contextmanager
def _ignoring_fear(viplanner: planner.ViPlanner):
    threshold = viplanner.fear_threshold
    viplanner.fear_threshold = 1.0
    try:
        yield
    finally:
        viplanner.fear_threshold = threshold


def _timed_step(features: dict) -> Step:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = planner.step(features)
    torch.cuda.synchronize()
    print(f"step {1000 * (time.perf_counter() - start):.0f} ms, fear {planner._planner.fear:.3f}")
    return result


def _chunk(result: Step) -> np.ndarray:
    assert isinstance(result, Step)
    assert not result.action and not result.command and not result.signal
    chunk = np.asarray(result.chunk)
    assert chunk.shape == (50, 2)
    assert np.isfinite(chunk).all()
    return chunk


def test_released_checkpoint_matches_its_training_config(viplanner: planner.ViPlanner):
    policy = viplanner.policy
    assert policy.train_cfg.sem and not policy.train_cfg.rgb
    assert policy.train_cfg.knodes == 5
    assert policy.input_size == (360, 640)
    assert policy.max_depth == 15.0
    assert policy.max_goal_distance == 15.0
    assert sum(p.numel() for p in policy.net.parameters()) == 71_045_584


def test_segmenter_paints_a_street_scene_in_viplanner_colors(viplanner: planner.ViPlanner):
    rgb = _street()
    torch.cuda.synchronize()
    start = time.perf_counter()
    colored = viplanner.segmenter(rgb)
    torch.cuda.synchronize()
    print(f"segmenter {1000 * (time.perf_counter() - start):.0f} ms on {rgb.shape[1]}x{rgb.shape[0]}")
    assert colored.shape == (3, rgb.shape[0], rgb.shape[1])
    pixels = (colored.permute(1, 2, 0) * 255).round().to(torch.uint8).reshape(-1, 3).cpu()
    palette = {tuple(color) for color in _COLORS.values()}
    present = {tuple(row) for row in np.unique(pixels.numpy(), axis=0).tolist()}
    assert present <= palette
    share = {
        name: (pixels == torch.tensor(color, dtype=torch.uint8)).all(dim=1).float().mean().item()
        for name, color in _COLORS.items()
    }
    print({name: round(value, 3) for name, value in share.items() if value > 0.0})
    assert share["road"] + share["sidewalk"] + share["floor"] > 0.2
    assert share["building"] + share["wall"] > 0.05


@pytest.mark.parametrize(("side", "goal_y"), [("left", 1.5), ("right", -1.5)])
def test_open_corridor_path_bends_toward_the_goal(
    viplanner: planner.ViPlanner, corridor_640x480: tuple[np.ndarray, np.ndarray], side: str, goal_y: float
):
    depth, semantic = corridor_640x480
    policy = viplanner.policy
    depth_in = planner.to_input(
        torch.from_numpy(planner.depth_meters(depth, policy.max_depth)).unsqueeze(0), policy.input_size
    )
    semantic_in = planner.to_input(torch.from_numpy(semantic).permute(2, 0, 1).float() / 255.0, policy.input_size)
    trajectory, fear = policy.predict(depth_in, semantic_in, np.array([5.0, goal_y, 0.0], dtype=np.float32))
    np.set_printoptions(precision=2, suppress=True)
    print(f"goal {side}: fear {fear:.3f}, keypoints\n{trajectory[10::10]}")
    assert fear < 0.7
    assert trajectory[-1, 0] > 2.0
    assert np.sign(trajectory[-1, 1]) == np.sign(goal_y)
    assert abs(trajectory[-1, 1]) > 0.3


def test_street_image_over_corridor_depth_trips_the_fear_stop(viplanner: planner.ViPlanner, corridor_640x480):
    depth, _ = corridor_640x480
    features = _street_features(depth)
    assert planner.step(features) == Step()
    assert viplanner.fear > viplanner.fear_threshold
    with _ignoring_fear(viplanner):
        _chunk(planner.step(features))


def test_full_step_latency_and_memory(viplanner: planner.ViPlanner, corridor_640x480):
    depth, _ = corridor_640x480
    features = _street_features(depth)
    with _ignoring_fear(viplanner):
        chunk = _chunk(_timed_step(features))
        print(f"chunk head (m):\n{chunk[:3]}\nchunk tail (m):\n{chunk[-3:]}")
        assert chunk[-1, 0] > 1.0
        latencies = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            planner.step(features)
            torch.cuda.synchronize()
            latencies.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"steady step {np.mean(latencies):.0f} ms (min {min(latencies):.0f}), peak {peak:.2f} GiB")


def test_millimeter_depth_gives_the_same_chunk(viplanner: planner.ViPlanner, corridor_640x480):
    depth, _ = corridor_640x480
    features = _street_features(depth, goal_y=1.0)
    millimeters = np.where(np.isfinite(depth), depth * 1000.0, 0.0).round().astype(np.uint16)
    with _ignoring_fear(viplanner):
        meters = _chunk(planner.step(features))
        from_mm = _chunk(planner.step({**features, "depth": millimeters}))
    np.testing.assert_allclose(from_mm, meters, atol=1e-2)


def test_same_observation_gives_same_chunk(viplanner: planner.ViPlanner, corridor_640x480):
    depth, _ = corridor_640x480
    features = _street_features(depth, goal_y=-1.0)
    with _ignoring_fear(viplanner):
        np.testing.assert_array_equal(_chunk(planner.step(features)), _chunk(planner.step(features)))


def test_square_camera_images_are_accepted(viplanner: planner.ViPlanner):
    depth, _ = corridor(256, 256)
    rgb = np.asarray(Image.open(_DATA / "street.jpg").convert("RGB").resize((224, 224)))
    planner.on_reset("square", {"goal_pose": {"x": 4.0, "y": 0.0, "theta": 0.0}})
    with _ignoring_fear(viplanner):
        chunk = _chunk(_timed_step({"robot_pose": np.zeros(3), "depth": depth, "rgb": rgb}))
    assert chunk[-1, 0] > 1.0


def test_no_goal_holds_still(viplanner: planner.ViPlanner, corridor_640x480):
    depth, _ = corridor_640x480
    planner.on_reset("nogoal", None)
    assert planner.step({"robot_pose": np.zeros(3), "depth": depth, "rgb": _street()}) == Step()


def test_missing_sensor_holds_still(viplanner: planner.ViPlanner, corridor_640x480):
    depth, _ = corridor_640x480
    planner.on_reset("nosensor", {"goal_pose": {"x": 4.0, "y": 0.0, "theta": 0.0}})
    assert planner.step({"robot_pose": np.zeros(3), "depth": depth, "rgb": None}) == Step()
    assert (
        planner.step({"robot_pose": np.zeros(3), "depth": np.empty((0,), dtype=np.uint8), "rgb": _street()}) == Step()
    )
