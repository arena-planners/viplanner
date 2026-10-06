# Attribution

## Code

Model code adapted from leggedrobotics/viplanner (commit 6fcf3c6, 2025-02-12).
Source: https://github.com/leggedrobotics/viplanner

`viplanner_model/` holds these upstream files verbatim, except that the absolute `viplanner.` imports became
`viplanner_model.` and `TrainCfg.data_cfg` takes `field(default_factory=DataCfg)`, as Python 3.11 and later refuse
the mutable default:

- `viplanner/plannernet/__init__.py`, `PlannerNet.py`, `autoencoder.py`, `rgb_encoder.py`
- `viplanner/traj_cost_opt/traj_opt.py`, with `__init__.py` cut to the `TrajOpt` and `CubicSplineTorch` exports
- `viplanner/config/learning_cfg.py`, `viplanner_sem_meta.py`, `coco_sem_meta.py`, with `__init__.py` cut to their
  exports (the cost-map configs and their dependencies are not vendored)
- `viplanner_model/__init__.py` keeps the upstream license header only

`planner.py` ports `ros/planner/src/vip_inference.py` (`VIPlannerInference.plan`: the resize to `img_input_size`, the
network call and `TrajGeneratorFromPFreeRot` at step 0.1), `ros/planner/src/viplanner_node.py` (`depthCallback`'s
depth scaling and zeroing, `goalProjection`'s crop to half the maximum goal distance and the camera offset, the path
transform into the robot frame, the 0.7 fear threshold) and `ros/planner/src/m2f_inference.py` (the COCO label to
ViPlanner color map through `get_class_for_id_mmdet` and `VIPlannerSemMetaHandler`). Mask2Former runs through
Hugging Face transformers (`facebook/mask2former-swin-tiny-coco-panoptic`, the detectron2 conversion of the
`model_final_9fd0ae.pkl` that `viplanner_node.py` defaults to) instead of mmdetection, and the labels come from the
per-pixel argmax (`post_process_semantic_segmentation`) rather than the panoptic map with unassigned pixels painted
`static`. The planner also center-crops both images to the 848x480 aspect of the training camera before the resize,
and holds still on a fear value above the threshold where upstream publishes the first waypoint only.

`tests/data/street.jpg` is panel A of upstream's `assets/crosswalk.jpg`.

## Weights

`model.pt` and `model.yaml` from the Google Drive links in upstream's README (checkpoint
`1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc`, config `1r1yhNQAJnjpn9-xpAQWGaQedwma5zokr`, both dated 2024-02-09), mirrored at
https://huggingface.co/arena-rosnav/viplanner under the upstream BSD-3-Clause license (clause 2, binary
redistribution with the notice, conditions and disclaimer in the model card). Mask2Former weights from
https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic (revision df6b114), the detectron2 conversion of
a model-zoo checkpoint of https://github.com/facebookresearch/Mask2Former (code MIT, weights CC BY-NC 4.0 per
`MODEL_ZOO.md`, non-commercial use with attribution, the model card's `license: other` resolves to it), fetched from
Facebook's Hugging Face repository without a mirror.

Runtime dependencies that are forks: none.

See LICENSE for upstream terms (BSD-3-Clause).

Cite:

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
