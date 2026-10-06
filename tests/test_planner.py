from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch
import yaml
from arena_planners import goal, weights
from arena_planners.bridge.chunk import check_chunk, chunk_config_from_manifest

import planner

_PLANNER_DIR = pathlib.Path(planner.__file__).parent
_NO_OFFSET = (0.0, 0.0)
_MAX_GOAL = 15.0


def _manifest() -> dict:
    with open(_PLANNER_DIR / "planner.yaml") as fh:
        return yaml.safe_load(fh)


def test_goal_in_camera_frame_rotates_by_heading():
    robot = np.array([1.0, 1.0, math.pi / 2])
    encoded = planner.goal_in_camera_frame(robot, {"x": 1.0, "y": 3.0, "theta": 0.0}, _NO_OFFSET, _MAX_GOAL)
    np.testing.assert_allclose(encoded, [2.0, 0.0, 0.0], atol=1e-6)


def test_goal_in_camera_frame_left_is_positive_y():
    encoded = planner.goal_in_camera_frame(np.zeros(3), {"x": 0.0, "y": 2.0, "theta": 0.0}, _NO_OFFSET, _MAX_GOAL)
    np.testing.assert_allclose(encoded, [0.0, 2.0, 0.0], atol=1e-6)


def test_goal_in_camera_frame_subtracts_camera_offset_after_cropping():
    encoded = planner.goal_in_camera_frame(np.zeros(3), {"x": 3.0, "y": 1.0, "theta": 0.0}, (0.03, 0.1), _MAX_GOAL)
    np.testing.assert_allclose(encoded, [2.97, 0.9, 0.0], atol=1e-6)
    far = planner.goal_in_camera_frame(np.zeros(3), {"x": 30.0, "y": 0.0, "theta": 0.0}, (0.03, 0.0), _MAX_GOAL)
    np.testing.assert_allclose(far, [7.5 - 0.03, 0.0, 0.0], atol=1e-6)


def test_goal_beyond_max_distance_is_pulled_to_half_of_it():
    encoded = planner.goal_in_camera_frame(np.zeros(3), {"x": 30.0, "y": 40.0, "theta": 0.0}, _NO_OFFSET, _MAX_GOAL)
    np.testing.assert_allclose(encoded, [4.5, 6.0, 0.0], atol=1e-5)
    inside = planner.goal_in_camera_frame(np.zeros(3), {"x": 9.0, "y": 12.0, "theta": 0.0}, _NO_OFFSET, _MAX_GOAL)
    np.testing.assert_allclose(inside, [9.0, 12.0, 0.0], atol=1e-5)


def test_goal_has_zero_height_and_float32_dtype():
    encoded = planner.goal_in_camera_frame(np.zeros(3), {"x": 1.0, "y": 0.0, "theta": 1.0}, _NO_OFFSET, _MAX_GOAL)
    assert encoded.dtype == np.float32
    assert encoded.shape == (3,)
    assert encoded[2] == 0.0


def test_depth_meters_from_millimeters_and_meters():
    mm = np.array([[500, 2000], [16000, 0]], dtype=np.uint16)
    np.testing.assert_allclose(planner.depth_meters(mm, 15.0), [[0.5, 2.0], [0.0, 0.0]])
    meters = np.array([[0.5, np.nan], [np.inf, 15.5]], dtype=np.float32)
    np.testing.assert_allclose(planner.depth_meters(meters, 15.0), [[0.5, 0.0], [0.0, 0.0]])
    assert planner.depth_meters(meters, 15.0).dtype == np.float32


def test_depth_meters_keeps_the_max_depth_itself():
    np.testing.assert_allclose(planner.depth_meters(np.array([[15.0]], dtype=np.float32), 15.0), [[15.0]])


@pytest.mark.parametrize(
    ("height", "width", "rows", "top"),
    [(480, 640, 362, 59), (256, 256, 145, 55), (360, 640, 360, 0), (480, 848, 480, 0), (720, 1280, 725, 0)],
)
def test_crop_keeps_the_training_aspect_centered(height: int, width: int, rows: int, top: int):
    image = torch.arange(height, dtype=torch.float32)[None, :, None].expand(1, height, width)
    cropped = planner.crop_to_training_aspect(image)
    assert cropped.shape == (1, min(rows, height), width)
    assert cropped[0, 0, 0].item() == top


def test_to_input_resizes_to_the_network_size_and_keeps_the_center_row():
    height, width = 480, 640
    image = torch.arange(height, dtype=torch.float32)[None, :, None].expand(1, height, width)
    out = planner.to_input(image, (360, 640))
    assert out.shape == (1, 1, 360, 640)
    assert out[0, 0, 180, 320].item() == pytest.approx(height / 2, abs=1.0)
    constant = planner.to_input(torch.full((3, 256, 256), 0.25), (360, 640))
    assert constant.shape == (1, 3, 360, 640)
    torch.testing.assert_close(constant, torch.full_like(constant, 0.25))


def test_chunk_drops_the_origin_and_adds_the_camera_offset():
    trajectory = np.zeros((51, 3))
    trajectory[:, 0] = np.linspace(0.0, 5.0, 51)
    trajectory[:, 1] = -np.linspace(0.0, 1.0, 51)
    trajectory[:, 2] = 0.3
    chunk = planner.to_chunk(trajectory, (0.03, 0.0))
    assert len(chunk) == 50
    assert all(len(point) == 2 for point in chunk)
    np.testing.assert_allclose(chunk[0], [0.13, -0.02])
    np.testing.assert_allclose(chunk[-1], [5.03, -1.0])
    assert check_chunk("waypoints", chunk) == chunk


def test_color_table_follows_upstream_class_mapping():
    names = ["person", "floor-wood", "wall-brick", "road", "chair", "door-stuff", "sky-other-merged", "dog"]
    table = planner.color_table(names)
    assert table.dtype == torch.uint8
    assert table.shape == (len(names) + 1, 3)
    expected = [
        [255, 0, 0],
        [0, 204, 0],
        [102, 0, 204],
        [255, 128, 0],
        [0, 0, 51],
        [153, 153, 0],
        [102, 0, 51],
        [204, 0, 0],
        [0, 0, 0],
    ]
    assert table.tolist() == expected


def test_manifest_is_a_pose_goal_waypoint_planner():
    manifest = _manifest()
    assert manifest["action_type"] == "waypoints"
    assert goal.declared(manifest) == frozenset({"pose"})
    assert manifest["signals"] == []
    config = chunk_config_from_manifest(manifest)
    assert config.execute_s == pytest.approx(1.0 / manifest["rate_hz"], abs=1e-3)
    datasources = manifest["observations"]["datasources"]
    assert datasources["robot_pose"]["type"] == "RobotPoseTFGenerator"
    assert datasources["rgb"]["params"] == {"sensor": "image"}
    assert datasources["depth"]["params"] == {"sensor": "depth"}
    assert set(manifest["config"]) == {"device", "camera_offset_x", "camera_offset_y", "fear_threshold"}


def test_weights_manifest_pins_checkpoint_and_segmenter():
    entries = weights.read(_PLANNER_DIR)
    by_dest = {entry.dest: entry for entry in entries}
    assert by_dest["model/model.pt"].repo == "arena-rosnav/viplanner"
    assert by_dest["model/model.yaml"].repo == "arena-rosnav/viplanner"
    segmenter = [entry for entry in entries if entry.dest.startswith("model/mask2former-swin-tiny-coco-panoptic/")]
    assert {pathlib.PurePosixPath(entry.dest).name for entry in segmenter} == {
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    }
    assert {entry.repo for entry in segmenter} == {"facebook/mask2former-swin-tiny-coco-panoptic"}
    assert {entry.revision for entry in segmenter} == {"df6b1142ff50c3276559d9d78f35f6a579c75a77"}
    assert all(len(entry.sha256) == 64 for entry in entries)


def test_steering_goal_prefers_the_plan_subgoal():
    goal_pose = {"x": 9.0, "y": 0.0, "theta": 0.0}
    assert planner.steering_goal({}, goal_pose) == goal_pose
    assert planner.steering_goal({"subgoal": None}, goal_pose) == goal_pose
    subgoal = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    assert planner.steering_goal({"subgoal": subgoal}, goal_pose) == {"x": 1.0, "y": 2.0, "theta": 0.5}


def test_manifest_steers_by_the_global_plan_subgoal():
    manifest = _manifest()
    assert manifest["depends"]["global_plan"] is True
    with pytest.raises(ValueError):
        goal.withhold_pose(manifest)
    sources = manifest["observations"]["datasources"]
    assert sources["subgoal"] == {"type": "SubgoalGenerator", "params": {"lookahead": 6.0}}
    assert sources["robot_pose_from_tf"]["type"] == "RobotPoseTFGenerator"
    assert sources["goal_pose"] == {"type": "geometry_msgs/PoseStamped", "params": {"topic": "goal_pose"}}
    assert sources["global_plan"] == {"type": "nav_msgs/Path", "params": {"topic": "plan"}}
    assert sources["costmap"] == {"type": "nav_msgs/OccupancyGrid", "params": {"topic": "global_costmap/costmap"}}
    names = list(sources)
    assert names.index("robot_pose_from_tf") < names.index("subgoal")
