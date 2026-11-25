# NERO integration

This branch is upstream RT-COSMIK (Gepetto/rt-cosmik) with the changes the NERO
retargeting workspace needs. The workspace pins one commit of it in
`external/perception.repos` and installs it editable through pixi.

RT-COSMIK stays transport-independent. ROS integration lives in the workspace's
adapter packages; nothing here depends on the NERO planner, controller,
simulator or visualization.

## Changes against upstream

- Model paths can be overridden with `RTCOSMIK_SAVE_DIR`, `RTCOSMIK_CANO_PATH`,
  `RTCOSMIK_NLF_PATH` and `RTCOSMIK_YOLO_PATH`. The detector defaults to the
  PyTorch weights (`weights/yolo/yolov10n.pt`), so no TensorRT engine is needed.
- Capture at 1920x1080. The marker low-pass cutoff stays at 5 Hz (upstream: 10 Hz).
  The fallback settings in `config_loader.py` follow the same values.
- `setup.py` declares no dependencies and copies no data; the environment provides
  the dependencies and the source tree provides config and weights.
- `PipelineProcess` can publish one `PipelineResult` per completed capture
  (`atomic_result_queue`): camera frame counters, capture timestamps, filtered
  markers and the matching joint configuration. `compute_human_kinematics=False`
  stops after the filtered markers and builds no solver, recorder or viewer.
  A protected hook gives an integration the same capture and its 2D keypoints.
- `vendor/` holds the custom `example_robot_data` human loader and the human
  URDF and meshes (Gepetto, BSD-2-Clause); the workspace adds it to `PYTHONPATH`.

## Binary artifacts

Neural weights, TensorRT/ONNX engines, sample data and videos stay untracked.
The workspace's model store links them into `weights/`. The canonical vertex
arrays are tracked upstream.

The branch `nero-7d60c04` keeps the previous integration on upstream 7d60c04.
