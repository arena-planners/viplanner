from __future__ import annotations

import numpy as np
import pytest
import torch

import planner
from viplanner_model.config import TrainCfg
from viplanner_model.plannernet import DualAutoEncoder
from viplanner_model.traj_cost_opt import TrajOpt

_CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def random_policy() -> planner.Policy:
    torch.manual_seed(0)
    return planner.Policy(DualAutoEncoder(TrainCfg()), TrainCfg(), _CPU)


def test_default_train_config_matches_the_released_checkpoint():
    cfg = TrainCfg()
    assert cfg.sem and not cfg.rgb
    assert cfg.knodes == 5
    assert cfg.in_channel == 16
    assert list(cfg.img_input_size) == [360, 640]
    assert cfg.data_cfg.max_depth == 15.0
    assert cfg.data_cfg.max_goal_distance == 15.0


def test_dual_autoencoder_predicts_five_keypoints_and_a_fear_probability(random_policy: planner.Policy):
    depth = torch.rand(1, 1, 360, 640)
    semantic = torch.rand(1, 3, 360, 640)
    with torch.inference_mode():
        keypoints, fear = random_policy.net(depth, semantic, torch.tensor([[3.0, 1.0, 0.0]]))
    assert keypoints.shape == (1, 5, 3)
    assert fear.shape == (1, 1)
    assert 0.0 <= fear.item() <= 1.0
    assert sum(p.numel() for p in random_policy.net.parameters()) == 71_045_584


def test_trajectory_passes_through_the_keypoints_at_tenth_steps():
    keypoints = torch.tensor([[[1.0, 0.2, 0.0], [2.0, 0.5, 0.0], [3.0, 0.4, 0.1], [4.0, 0.0, 0.0], [5.0, -0.5, 0.0]]])
    trajectory = TrajOpt().TrajGeneratorFromPFreeRot(keypoints, step=0.1)
    assert trajectory.shape == (1, 51, 3)
    torch.testing.assert_close(trajectory[0, 0], torch.zeros(3))
    torch.testing.assert_close(trajectory[0, 10::10], keypoints[0], atol=1e-5, rtol=0.0)


def test_policy_predict_returns_a_camera_frame_path_and_a_float_fear(random_policy: planner.Policy):
    depth = planner.to_input(torch.full((1, 480, 640), 3.0), random_policy.input_size)
    semantic = planner.to_input(torch.zeros(3, 480, 640), random_policy.input_size)
    trajectory, fear = random_policy.predict(depth, semantic, np.array([3.0, 0.0, 0.0], dtype=np.float32))
    assert trajectory.shape == (51, 3)
    assert np.isfinite(trajectory).all()
    np.testing.assert_allclose(trajectory[0], 0.0)
    assert isinstance(fear, float)
    assert len(planner.to_chunk(trajectory, (0.03, 0.0))) == 50


def test_corridor_fixture_has_floor_below_the_horizon(corridor_640x480: tuple[np.ndarray, np.ndarray]):
    depth, semantic = corridor_640x480
    assert depth.shape == (480, 640) and semantic.shape == (480, 640, 3)
    assert depth[479, 320] == pytest.approx(0.5 / (239.5 / 320), rel=1e-3)
    assert depth[240, 320] == pytest.approx(8.0)
    assert semantic[479, 320].tolist() == [0, 204, 0]
    assert semantic[240, 0].tolist() == [102, 0, 204]
    assert semantic[0, 320].tolist() == [25, 0, 51]
