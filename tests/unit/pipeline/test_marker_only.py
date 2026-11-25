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
        poses = [torch.zeros((1, 2, 3)) for _ in frames]
        return {"poses2d": poses}, 0.0, None, None


class _IdentityIIR:
    def __init__(self, **_kwargs):
        pass

    def add_filter(self, **_kwargs):
        pass

    def filter(self, values):
        return values


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
        N=1,
        order=2,
        cutoff_freq=5.0,
        filter_type="lowpass",
        human_height=1.8,
        human_weight=75.0,
        human_gender="m",
        ik_type="sbs",
        keys_to_track_list=["LEFT", "RIGHT"],
        dt=0.025,
    )


def _timestamp_buffer(value):
    encoded = value.encode()
    return bytearray(encoded + b"\0" * (64 - len(encoded)))


def _pipeline_inputs(monkeypatch, *, iterations):
    monkeypatch.setattr(pipeline_module, "NLFEstimator", _FakeEstimator)
    monkeypatch.setattr(pipeline_module, "IIR", _IdentityIIR)

    marker_positions = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )
    monkeypatch.setattr(
        pipeline_module,
        "triangulate_points",
        lambda **_kwargs: marker_positions.copy(),
    )

    counters = [SimpleNamespace(value=1), SimpleNamespace(value=1)]
    frame_shape = (1, 1, 3)
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
        frame_shape=frame_shape,
        num_cameras=2,
    ), marker_positions


def _fail_if_called(name):
    def fail(*_args, **_kwargs):
        raise AssertionError(f"marker-only execution called {name}")

    return fail


def test_marker_only_rejects_legacy_result_queues_immediately(monkeypatch):
    inputs, _ = _pipeline_inputs(monkeypatch, iterations=1)

    with pytest.raises(ValueError, match="legacy results_queues"):
        PipelineProcess(
            **inputs,
            results_queues=[Queue(), Queue()],
            atomic_result_queue=Queue(),
            compute_human_kinematics=False,
        )


def test_marker_only_emits_first_filtered_capture_without_human_model(monkeypatch):
    inputs, expected_markers = _pipeline_inputs(monkeypatch, iterations=1)
    monkeypatch.setattr(
        pipeline_module.robex.human,
        "HumanLoader",
        _fail_if_called("HumanLoader"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "scale_human_model",
        _fail_if_called("scale_human_model"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "mks_registration",
        _fail_if_called("mks_registration"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "recalibrate_marker_frames_in_joint_space",
        _fail_if_called("recalibrate_marker_frames_in_joint_space"),
    )
    monkeypatch.setattr(pipeline_module, "RT_IK", _fail_if_called("RT_IK"))
    monkeypatch.setattr(pipeline_module, "RT_SWIKA", _fail_if_called("RT_SWIKA"))

    result_queue = Queue()
    pipeline = _RecordingPipeline(
        **inputs,
        results_queues=None,
        atomic_result_queue=result_queue,
        compute_human_kinematics=False,
    )

    pipeline.run()

    result = result_queue.get_nowait()
    assert result.camera_frame_counters == (1, 1)
    assert result.camera_capture_timestamps == (
        "camera-0-time",
        "camera-1-time",
    )
    assert tuple(result.markers) == ("LEFT", "RIGHT")
    np.testing.assert_allclose(result.markers["LEFT"], expected_markers[0])
    np.testing.assert_allclose(result.markers["RIGHT"], expected_markers[1])
    assert result.q is None
    assert pipeline.first_sample is False

    assert len(pipeline.completed) == 1
    completed_result, completed_frames, completed_keypoints = pipeline.completed[0]
    assert completed_result is result
    assert len(completed_frames) == 2
    assert len(completed_keypoints) == 2
    with pytest.raises(Empty):
        result_queue.get_nowait()


def test_default_full_mode_preserves_calibration_then_ik_emission(monkeypatch):
    inputs, _ = _pipeline_inputs(monkeypatch, iterations=2)

    human_model = SimpleNamespace()
    human_loader = SimpleNamespace(robot=SimpleNamespace(model=human_model))
    monkeypatch.setattr(
        pipeline_module.robex.human,
        "HumanLoader",
        lambda **_kwargs: human_loader,
    )
    monkeypatch.setattr(
        pipeline_module,
        "scale_human_model",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        pipeline_module,
        "mks_registration",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        pipeline_module,
        "recalibrate_marker_frames_in_joint_space",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        pipeline_module.pin,
        "neutral",
        lambda _model: np.zeros(2),
    )

    class FakeIK:
        def __init__(self, *_args, **_kwargs):
            self._q0 = None
            self._dict_m = None

        def solve_ik_sample_casadi(self):
            return np.array([1.0, 2.0])

        def solve_ik_sample_quadprog(self):
            return np.array([3.0, 4.0])

    monkeypatch.setattr(pipeline_module, "RT_IK", FakeIK)

    result_queue = Queue()
    pipeline = _RecordingPipeline(
        **inputs,
        results_queues=None,
        atomic_result_queue=result_queue,
    )

    assert pipeline.compute_human_kinematics is True
    pipeline.run()

    result = result_queue.get_nowait()
    assert result.camera_frame_counters == (2, 2)
    np.testing.assert_allclose(result.q, [3.0, 4.0])
    assert len(pipeline.completed) == 1
    assert pipeline.completed[0][0] is result
    with pytest.raises(Empty):
        result_queue.get_nowait()


def test_default_full_mode_still_accepts_legacy_double_queues(monkeypatch):
    inputs, _ = _pipeline_inputs(monkeypatch, iterations=1)

    pipeline = PipelineProcess(
        **inputs,
        results_queues=[Queue(), Queue()],
    )

    assert pipeline.compute_human_kinematics is True
