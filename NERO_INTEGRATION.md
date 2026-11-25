# RT-COSMIK source integration

This directory contains the ROS-free RT-COSMIK source imported from
`/workspace/cosmik_2_pixi/RT-COSMIK` on 2026-07-15. The accompanying custom
`example_robot_data` human loader and model assets were imported from that
workspace's `vendor/` directory.

The NERO workspace intentionally keeps RT-COSMIK transport-independent. ROS
integration belongs in a separate adapter package; RT-COSMIK itself must not
depend on the NERO planner, controller, Gazebo, or visualization consumers.

## Local integration changes

- Runtime paths are resolved relative to this source directory rather than a
  historical `/root/workspace/RT-COSMIK` checkout.
- `scripts/bash/fetch_models.sh` resolves this nested source directory instead
  of the outer Git repository root.
- The custom human model loader and its data live under `vendor/` and are added
  to `PYTHONPATH` by the workspace's `cosmik` Pixi feature.
- `PipelineProcess` has one optional atomic result queue carrying camera
  counters, capture timestamps, markers, and an optional matching IK
  configuration.
- One optional computation flag lets an atomic-result consumer stop after the
  existing filtered markers. Full human kinematics remains the default, and
  the original two viewer queues remain full-only for upstream scripts.
- The marker-only capability contains no NERO or ROS policy. NERO selects it in
  its adapter; perception, triangulation, world conversion, and filtering are
  shared with full mode.

## Binary artifacts

Large neural weights, generated TensorRT/ONNX artifacts, sample camera data,
and test videos remain untracked. A developer's checkout may contain these
files locally, but they must not be committed to the NERO Git history. The
small canonical vertex arrays and custom human URDF/meshes are tracked because
they are required runtime model data and are not generated build products.
