from queue import Empty, Queue
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rtcosmik.pipeline.pipeline as pipeline_module
from rtcosmik.pipeline.pipeline import PipelineProcess


class _IterationStop:
    def __init__(self, counters, iterations):
        self._counters = counters
        self._iterations = iterations
        self._checks = 0

    def is_set(self):
        if self._checks >= self._iterations:
            return True
        if self._checks:
            for counter in self._counters:
                counter.value += 1
        self._checks += 1
        return False


class _FakeEstimator:
    def __init__(self, **_kwargs):
        pass

    def estimate_from_frames(self, frames):
        poses2d = [torch.zeros((1, 2, 2)) for _ in frames]
        poses3d = [torch.zeros((1, 2, 3)) for _ in frames]
        return {"poses2d": poses2d, "poses3d": poses3d}, {}, None, None


class _IdentityFilter:
    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, frame):
        return np.asarray(frame)


class _RecordingPipeline(PipelineProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed = []

    def _on_completed_result(self, result, frames, keypoints_list):
        self.completed.append((result, frames, keypoints_list))


def _settings():
    return SimpleNamespace(
        yolo_path="unused-yolo",
        nlf_path="unused-nlf",
        cano_path="unused-canonical-vertices",
        nlf_indices=[0, 1],
        yolo_conf=0.2,
        yolo_imgsz=64,
        device="cpu",
        marker_names=["LEFT", "RIGHT"],
        fs=40,
        order=2,
        cutoff_freq=5.0,
        filter_type="lowpass",
    )


def _timestamp_buffer(value):
    encoded = value.encode()
    return bytearray(encoded + b"\0" * (64 - len(encoded)))


def _fail_if_called(name):
    def fail(*_args, **_kwargs):
        raise AssertionError(f"marker-only execution called {name}")

    return fail


def _pipeline_inputs(monkeypatch, *, iterations):
    monkeypatch.setattr(pipeline_module, "NLFEstimator", _FakeEstimator)
    monkeypatch.setattr(pipeline_module, "MarkerFilter", _IdentityFilter)
    monkeypatch.setattr(pipeline_module, "resolve_detector_engine", lambda path, _n: path)

    marker_positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    monkeypatch.setattr(
        pipeline_module, "reconstruct_3d", lambda *_args, **_kwargs: marker_positions.copy()
    )

    counters = [SimpleNamespace(value=1), SimpleNamespace(value=1)]
    return dict(
        settings=_settings(),
        frame_counters=counters,
        camera_buffers=[bytearray(3), bytearray(3)],
        camera_locks=[Lock(), Lock()],
        timestamp_buffers=[
            _timestamp_buffer("camera-0-time"),
            _timestamp_buffer("camera-1-time"),
        ],
        stop_event=_IterationStop(counters, iterations),
        mtxs=[np.eye(3), np.eye(3)],
        dists=[np.zeros(5), np.zeros(5)],
        projections=[np.zeros((3, 4)), np.zeros((3, 4))],
        world_R1_cam=np.eye(3),
        world_T1_cam=np.zeros(3),
        frame_shape=(1, 1, 3),
        num_cameras=2,
    ), marker_positions


def test_marker_only_requires_the_atomic_result_queue(monkeypatch):
    inputs, _ = _pipeline_inputs(monkeypatch, iterations=1)

    with pytest.raises(ValueError, match="atomic result queue"):
        PipelineProcess(**inputs, compute_human_kinematics=False)


def test_marker_only_emits_first_filtered_capture_without_human_model(monkeypatch):
    inputs, expected_markers = _pipeline_inputs(monkeypatch, iterations=1)
    for name in ("HumanSolver", "Recorder", "AsyncDisplay", "Viewer"):
        monkeypatch.setattr(pipeline_module, name, _fail_if_called(name))

    result_queue = Queue()
    pipeline = _RecordingPipeline(
        **inputs, atomic_result_queue=result_queue, compute_human_kinematics=False
    )
    pipeline.run()

    result = result_queue.get_nowait()
    assert result.camera_frame_counters == (1, 1)
    assert result.camera_capture_timestamps == ("camera-0-time", "camera-1-time")
    assert tuple(result.markers) == ("LEFT", "RIGHT")
    np.testing.assert_allclose(result.markers["LEFT"], expected_markers[0])
    np.testing.assert_allclose(result.markers["RIGHT"], expected_markers[1])
    assert result.q is None

    assert len(pipeline.completed) == 1
    completed_result, completed_frames, completed_keypoints = pipeline.completed[0]
    assert completed_result is result
    assert len(completed_frames) == 2
    assert len(completed_keypoints) == 2
    assert completed_keypoints[0].shape == (2, 2)
    with pytest.raises(Empty):
        result_queue.get_nowait()


def test_default_full_mode_emits_after_calibration_with_ik(monkeypatch):
    inputs, _ = _pipeline_inputs(monkeypatch, iterations=2)

    class FakeSolver:
        model = collision_model = visual_model = None

        def __init__(self, *_args, **_kwargs):
            pass

        def calibrate(self, _markers):
            return np.array([1.0, 2.0])

        def step(self, _markers):
            return np.array([3.0, 4.0])

    class FakeRecorder:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            return self

        def record(self, *_args):
            pass

        def close(self):
            pass

    class FakeDisplay:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def submit(self, _job):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pipeline_module, "HumanSolver", FakeSolver)
    monkeypatch.setattr(pipeline_module, "Recorder", FakeRecorder)
    monkeypatch.setattr(pipeline_module, "AsyncDisplay", FakeDisplay)
    monkeypatch.setattr(pipeline_module, "Viewer", lambda *_args: None)

    result_queue = Queue()
    pipeline = _RecordingPipeline(**inputs, atomic_result_queue=result_queue)
    assert pipeline.compute_human_kinematics is True
    pipeline.run()

    result = result_queue.get_nowait()
    assert result.camera_frame_counters == (2, 2)
    np.testing.assert_allclose(result.q, [3.0, 4.0])
    assert len(pipeline.completed) == 1
    assert pipeline.completed[0][0] is result
    with pytest.raises(Empty):
        result_queue.get_nowait()


def test_default_full_mode_needs_no_result_queue(monkeypatch):
    inputs, _ = _pipeline_inputs(monkeypatch, iterations=1)
    assert PipelineProcess(**inputs).atomic_result_queue is None
