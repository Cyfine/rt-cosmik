"""Transport-neutral output from one completed RT-COSMIK pipeline step."""

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class PipelineResult:
    """Markers and optional IK state from one synchronized camera bundle."""

    camera_frame_counters: Tuple[int, ...]
    camera_capture_timestamps: Tuple[str, ...]
    markers: Mapping[str, np.ndarray]
    q: np.ndarray | None
