from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
readme = ROOT / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

# Kept intentionally minimal.
#   - The heavy deps (torch, pinocchio+casadi, example-robot-data, opencv, ...)
#     are provided by the environment (pixi: conda-forge + pypi), NOT pinned here,
#     so the editable install never pulls the casadi-less pip `pin` wheel.
#   - config/weights are located at runtime via rtcosmik.config_loader walking the
#     source tree, so there is no data_files copy (avoids duplicating ~800MB).
setup(
    name="rt-cosmik",
    version="0.1.0",
    description="Real-Time Constrained and Open-Source Multibody Inverse Kinematics (core engine, ROS-free)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
