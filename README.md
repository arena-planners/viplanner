# viplanner

[ViPlanner](https://leggedrobotics.github.io/viplanner.github.io/) (Roth et al., ICRA 2024, arXiv 2310.00982) behind the
arena_planners bridge: a local planner trained by imperative learning on depth and semantic images. From the depth
image, a class-color semantic image and a goal point in the camera frame it predicts five keypoints and a fear value
(the probability that the path collides), and the cubic-spline path through the keypoints is returned as a
51-point waypoint chunk at 5 Hz. The released checkpoint takes semantics, so a Mask2Former (Swin-T, COCO panoptic,
the default of upstream's ROS node) runs in front of it and its COCO labels are recolored into ViPlanner's class
palette.

```sh
arena feature planners add viplanner
arena launch sim:=gazebo robot:=jackal[camera=vln_rgbd] robot.planner:=viplanner
```

`robot.planner:=viplanner` selects `robot.mobile:=vla`. The planner takes the goal pose only (`goal_inputs: [pose]`),
so `robot.mobile.goal:=pose` is also the default. The robot needs an `image` and a `depth` sensor in
`model_params.yaml`, both at 90 degrees HFOV with square pixels. The weights take 0.5 GB on disk and 0.8 GB of GPU
memory.

## Inputs

| Datasource | Topic (jackal `camera=vln_rgbd`) | Model input |
|---|---|---|
| `depth` (`depth` sensor) | `vln_camera/depth_image`, 256x256 `32FC1` | meters (`16UC1` divided by 1000), NaN, inf and values over 15 m set to 0, center rows cropped to the 848x480 training aspect, resized to 360x640 |
| `rgb` (`image` sensor) | `vln_camera/image`, 224x224 `rgb8` | Mask2Former at the camera resolution, per-pixel argmax over the 133 COCO panoptic labels, each label painted with its ViPlanner class color, cropped and resized like the depth image, values in [0, 1] |
| goal | from the subgoal and `robot_pose` | `[x, y, 0]` in the camera frame (x forward, y left), pulled to 7.5 m along the same bearing when farther than 15 m |
| `subgoal` | `plan` from the nav2 planner server, `global_costmap/costmap` | the farthest point up to 6 m along the plan whose straight chord from the robot stays clear of the costmap, used in place of the goal pose, the goal pose itself until a plan arrives |

`depends.global_plan: true` makes the adapter start a nav2 planner server (`robot.mobile.global_planner`, default
`nav2/navfn`) that replans to the goal every second, and the `SubgoalGenerator` datasource turns the plan into
`subgoal`. The model steers at that point, so it only ever sees a goal in line of sight. A goal pose without a route
left the robot stuck behind walls, see Notes.

The crop keeps `round(width * 480 / 848)` rows around the image center, 145 of the 256 depth rows and 127 of the 224
RGB rows on the jackal camera. The COCO-to-ViPlanner mapping is upstream's `get_class_for_id_mmdet`, substring matching
included (`cardboard` lands on `vehicle`, `railroad` on `road`, `cabinet-merged` on `dynamic`). Upstream's node paints
the panoptic map and leaves unassigned pixels black (`static`, an obstacle class). On the street image in `tests/data`
the panoptic fusion leaves between 14 and 72 percent of the frame unassigned depending on the input size, so this
planner takes the dense argmax labels of the same network instead.

### Camera

The checkpoint (`model.yaml`, `env_list` of `*_cam_mount` Matterport scenes, CARLA town01 and the warehouse) was trained
on a depth camera of 848x480 and a semantic camera of 1280x720, both at the RealSense D435 optics of upstream's CARLA
config (focal length 1.93 mm, horizontal aperture 3.8 mm, 89 degrees HFOV, 58 degrees VFOV), mounted 0.51 m ahead of
ANYmal's base at the base height of about 0.5 m above the floor, level, and resized to 360x640 for the network. The
jackal `vln_rgbd` camera is 224x224 RGB and 256x256 depth at 90 degrees HFOV, 0.98 m above the floor, level
(`arena_robots/arena_robots/components/camera/vln_rgbd/vln_rgbd.urdf.xacro`). The HFOV matches, and the crop above gives
the training VFOV of 59 degrees, so the geometry is right. Two things differ: the network input is upscaled 2.5 times
from a quarter of the training resolution, and the camera sits 0.5 m higher than in training, which moves the floor
and obstacle rows in both images. A dedicated variant with 848x480 depth and RGB at 90 degrees HFOV mounted 0.5 m above
the floor would remove both differences. The planner runs with the present camera.

## Actions

Each tick the model predicts five keypoints in the camera frame at increasing distance toward the goal, with the
spline path sampled at a tenth of the keypoint spacing (51 points from the origin). The planner drops the origin,
moves the points by the camera offset into the robot frame and returns the 50 remaining `[x, y]` waypoints. The bridge
follows them with pure pursuit (lookahead 0.5 m, 1.0 m/s, capped by the robot's limits) for one tick of 0.2 s, then asks
again. Upstream's ANYmal path follower uses a 0.25 m lookahead and a 1.5 m/s speed cap, and its node plans at 5 Hz.

A fear value above `fear_threshold` returns an empty chunk, so the robot holds still until a later observation plans a
path below the threshold. Upstream's node uses the same 0.7 entry threshold for its fear reaction, which publishes only
the first waypoint, although its buffer counter never exceeds zero at the pinned commit. Without a goal (a reset before
the goal is dispatched) or without a depth or RGB frame the planner holds still as well.

The model has no arrival output, so the planner lists no signals and goto phases with `signal: arrived` refuse it.
The goal pose decides success.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `device` | `cuda` | Torch device for both networks. |
| `camera_offset_x` | `0.03` | Camera position ahead of the robot base in meters (the jackal mast puts it 0.03 m forward). |
| `camera_offset_y` | `0.0` | Camera position left of the robot base in meters. |
| `fear_threshold` | `0.7` | Fear value above which the planner holds still. `1.0` follows every path. |

## Weights and license

`arena feature planners add viplanner` fetches `model/model.pt` and `model/model.yaml`, the checkpoint and training
config linked from upstream's README (Google Drive), mirrored at `arena-rosnav/viplanner` on Hugging Face, and the
Mask2Former files from `facebook/mask2former-swin-tiny-coco-panoptic` (revision `df6b114`). Every file is pinned by
sha256.

The ViPlanner code and the checkpoint are BSD-3-Clause (upstream `LICENSE`, Pascal Roth, Robotic Systems Lab, ETH
Zurich). The checkpoint ships without a license of its own and is published through the README of that repository, so
the same license applies, and clause 2 permits redistribution in binary form when the copyright notice, the conditions
and the disclaimer accompany it, which the mirror's model card does. Upstream's README also says "All right reserved"
under its own License heading, the `LICENSE` file is the operative grant. The Mask2Former code is MIT
(facebookresearch/Mask2Former) and its weights are CC BY-NC 4.0 (`MODEL_ZOO.md`, License section: every model offered
there, of which the Hugging Face checkpoint is the detectron2 conversion), non-commercial use with attribution. The
model card's `license: other` resolves to that license. The weights are fetched from Facebook's Hugging Face
repository and not mirrored. Cite Roth et al. 2024 when using this planner:

```bibtex
@inproceedings{roth2024viplanner,
  title={ViPlanner: Visual Semantic Imperative Learning for Local Navigation},
  author={Roth, Pascal and Nubert, Julian and Yang, Fan and Mittal, Mayank and Hutter, Marco},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5243--5249},
  year={2024},
  organization={IEEE}
}
```

## Notes

- Measured on an RTX PRO 6000 Blackwell with torch 2.14.1+cu130 and transformers 4.57.6: 0.5 s to load, 0.44 GiB
  allocated after load, 0.83 GiB peak, 15 to 17 ms per step on 640x480 inputs and 14 ms on the 224x224 and 256x256
  jackal inputs, after a first segmenter call of about 0.3 s.
- The route decides whether a goal behind walls is reached. On ten start and goal pairs of hospital_1 (the five
  `hospital_complex` stages and five more from shifted seeds, 4 to 24 m apart, 2 to 8 pedestrians, 100 s and one
  episode per pair) the planner reached 6 goals steering by the subgoal and 3 steering by the goal pose.
- `tests/test_model.py` needs the weights and a CUDA device, the other tests run on CPU without them. The synthetic
  corridor in `tests/conftest.py` gives a fear of 0.03 to 0.06 and a path that bends toward the goal, and the street
  photo over that corridor depth gives a fear of 0.83, on which the tests check the hold.
- The vendored `learning_cfg.py` takes `field(default_factory=DataCfg)` for `data_cfg`, as Python 3.11 and later
  refuse the mutable dataclass default of upstream.
