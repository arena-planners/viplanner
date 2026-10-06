"""ViPlanner wrapper for the arena_planners bridge."""

from __future__ import annotations

import logging
import math
import pathlib

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from arena_planners.sdk import Step, load_manifest, main_loop
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from viplanner_model.config import TrainCfg, VIPlannerSemMetaHandler, get_class_for_id_mmdet
from viplanner_model.plannernet import DualAutoEncoder
from viplanner_model.traj_cost_opt import TrajOpt

_DIR = pathlib.Path(__file__).parent
MODEL_DIR = _DIR / "model"
SEGMENTER_DIR = MODEL_DIR / "mask2former-swin-tiny-coco-panoptic"
_FETCH_HINT = "run `arena feature planners add viplanner` (or `python -m arena_planners fetch viplanner`)"

_TRAJ_STEP = 0.1
_DEPTH_MM_PER_M = 1000.0
_TRAIN_CAMERA_WIDTH = 848
_TRAIN_CAMERA_HEIGHT = 480

_log = logging.getLogger("viplanner")


def steering_goal(features: dict, goal: dict) -> dict:
    """The global plan's lookahead point when the bridge sends one as `subgoal`, else the episode goal."""
    subgoal = features.get("subgoal")
    if subgoal is None:
        return goal
    x, y, theta = (float(v) for v in np.asarray(subgoal, dtype=np.float64)[:3])
    return {"x": x, "y": y, "theta": theta}


def goal_in_camera_frame(
    robot: np.ndarray, goal: dict, camera_offset: tuple[float, float], max_goal_distance: float
) -> np.ndarray:
    """Upstream's `goalProjection`: goal `[x, y, 0]` in the level camera frame, pulled to half `max_goal_distance`."""
    rx, ry, rtheta = (float(v) for v in robot[:3])
    dx, dy = goal["x"] - rx, goal["y"] - ry
    fwd = dx * math.cos(rtheta) + dy * math.sin(rtheta)
    left = -dx * math.sin(rtheta) + dy * math.cos(rtheta)
    distance = math.hypot(fwd, left)
    if distance > max_goal_distance:
        fwd, left = fwd / distance * max_goal_distance / 2, left / distance * max_goal_distance / 2
    return np.array([fwd - camera_offset[0], left - camera_offset[1], 0.0], dtype=np.float32)


def depth_meters(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Upstream's `depthCallback`: meters from `32FC1` or `16UC1`, non-finite and beyond-`max_depth` values zeroed."""
    arr = np.asarray(depth)
    meters = arr.astype(np.float32) / _DEPTH_MM_PER_M if arr.dtype == np.uint16 else arr.astype(np.float32)
    meters[~np.isfinite(meters)] = 0.0
    meters[meters > max_depth] = 0.0
    return meters


def crop_to_training_aspect(image: torch.Tensor) -> torch.Tensor:
    """Center rows of a `(C, H, W)` image so that H/W is the 848x480 of the training camera, the width kept."""
    height, width = image.shape[-2:]
    keep = min(height, round(width * _TRAIN_CAMERA_HEIGHT / _TRAIN_CAMERA_WIDTH))
    top = (height - keep) // 2
    return image[..., top : top + keep, :]


def to_input(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """`(1, C, *size)` network input: the training aspect crop, resized like upstream's `img_converter`."""
    return tvf.resize(crop_to_training_aspect(image), list(size), antialias=True).unsqueeze(0)


def to_chunk(trajectory: np.ndarray, camera_offset: tuple[float, float]) -> list[list[float]]:
    """Waypoints `[x, y]` in the robot frame from the `(N, 3)` camera-frame path, its origin dropped."""
    points = np.asarray(trajectory, dtype=np.float64)[1:, :2] + np.asarray(camera_offset, dtype=np.float64)
    return points.tolist()


def color_table(class_names: list[str]) -> torch.Tensor:
    """`(N + 1, 3)` uint8 ViPlanner colors per COCO label, upstream's `coco_viplanner_color_mapping`, row N `static`."""
    meta = VIPlannerSemMetaHandler()
    mapping = get_class_for_id_mmdet(class_names)
    static = meta.class_color["static"]
    rows = [meta.class_color[mapping[i]] if i in mapping else static for i in range(len(class_names))]
    return torch.tensor([*rows, static], dtype=torch.uint8)


class Segmenter:
    """Mask2Former (Swin-T, COCO panoptic) painted in ViPlanner's class colors, upstream's `Mask2FormerInference`."""

    def __init__(self, model_dir: pathlib.Path, device: torch.device) -> None:
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"viplanner segmenter weights missing at {model_dir}, {_FETCH_HINT}")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device).eval()
        id2label = self.model.config.id2label
        self.colors = color_table([id2label[i] for i in range(len(id2label))]).to(device)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        """Class-color image `(3, H, W)` in [0, 1] on the device from the per-pixel argmax over the COCO labels."""
        height, width = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        labels = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
        return self.colors[labels.to(self.device).long()].permute(2, 0, 1).float() / 255.0


class Policy:
    """ViPlanner's dual-encoder PlannerNet and cubic-spline path generator, upstream's `VIPlannerInference`."""

    def __init__(self, net: DualAutoEncoder, train_cfg: TrainCfg, device: torch.device) -> None:
        if not train_cfg.sem or train_cfg.rgb:
            raise ValueError("viplanner: the checkpoint config is not the depth and semantics variant")
        data_cfg = train_cfg.data_cfg[0] if isinstance(train_cfg.data_cfg, list) else train_cfg.data_cfg
        self.train_cfg = train_cfg
        self.max_depth = float(data_cfg.max_depth)
        self.max_goal_distance = float(data_cfg.max_goal_distance)
        self.input_size = (int(train_cfg.img_input_size[0]), int(train_cfg.img_input_size[1]))
        self.device = device
        self.net = net.to(device).eval()
        self.traj = TrajOpt()

    @torch.inference_mode()
    def predict(self, depth: torch.Tensor, semantic: torch.Tensor, goal: np.ndarray) -> tuple[np.ndarray, float]:
        """Camera-frame path `(51, 3)` from the origin through the 5 keypoints, and the fear value."""
        goal_t = torch.from_numpy(np.asarray(goal, dtype=np.float32)).unsqueeze(0).to(self.device)
        keypoints, fear = self.net(depth.to(self.device), semantic.to(self.device), goal_t)
        trajectory = self.traj.TrajGeneratorFromPFreeRot(keypoints, step=_TRAJ_STEP)
        return trajectory[0].cpu().numpy(), float(fear.item())


def load_policy(model_dir: pathlib.Path, device: torch.device) -> Policy:
    """The released checkpoint (`model.pt` and its `model.yaml` training config) loaded strictly."""
    config_path, weights_path = model_dir / "model.yaml", model_dir / "model.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"viplanner checkpoint missing at {model_dir}, {_FETCH_HINT}")
    train_cfg = TrainCfg.from_yaml(str(config_path))
    net = DualAutoEncoder(train_cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state[0] if isinstance(state, (tuple, list)) else state, strict=True)
    return Policy(net, train_cfg, device)


class ViPlanner:
    def __init__(
        self, policy: Policy, segmenter: Segmenter, camera_offset: tuple[float, float], fear_threshold: float
    ) -> None:
        self.policy = policy
        self.segmenter = segmenter
        self.camera_offset = camera_offset
        self.fear_threshold = fear_threshold
        self.fear = 0.0
        self._warned: set[str] = set()
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        self.goal: dict | None = (initial_state or {}).get("goal_pose")
        self.fear = 0.0

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log.warning(msg)

    def inputs(self, depth: np.ndarray, rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Depth in meters and the class-color image, each `(1, C, 360, 640)`."""
        size = self.policy.input_size
        depth_t = torch.from_numpy(depth_meters(depth, self.policy.max_depth)).unsqueeze(0)
        return to_input(depth_t, size), to_input(self.segmenter(np.ascontiguousarray(rgb[..., :3])), size)

    def step(self, features: dict) -> Step:
        if self.goal is None:
            return Step()
        robot_pose, depth, rgb = features.get("robot_pose"), features.get("depth"), features.get("rgb")
        if (
            robot_pose is None
            or depth is None
            or rgb is None
            or np.asarray(depth).ndim != 2
            or np.asarray(rgb).ndim != 3
        ):
            self._warn_once("sensors", "viplanner: robot pose, depth or rgb missing, standing still")
            return Step()
        goal = goal_in_camera_frame(
            np.asarray(robot_pose),
            steering_goal(features, self.goal),
            self.camera_offset,
            self.policy.max_goal_distance,
        )
        depth_in, semantic_in = self.inputs(np.asarray(depth), np.asarray(rgb))
        trajectory, self.fear = self.policy.predict(depth_in, semantic_in, goal)
        if self.fear > self.fear_threshold:
            return Step()
        return Step(chunk=to_chunk(trajectory, self.camera_offset))


_planner: ViPlanner | None = None


def on_init(config: dict) -> None:
    global _planner
    device = torch.device(str(config.get("device", "cuda")))
    camera_offset = (float(config.get("camera_offset_x", 0.0)), float(config.get("camera_offset_y", 0.0)))
    fear_threshold = float(config.get("fear_threshold", 0.7))
    _planner = ViPlanner(
        load_policy(MODEL_DIR, device), Segmenter(SEGMENTER_DIR, device), camera_offset, fear_threshold
    )
    _log.warning("viplanner: loaded on %s", device)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
