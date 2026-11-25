import os

# Vendored layout: <repo>/vendor/example_robot_data/path.py  ->  <repo>/vendor/erd_data/robots
_VENDORED_ROBOTS = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "erd_data", "example-robot-data", "robots",
    )
)

# Honor an explicit override, otherwise fall back to the vendored data dir
# (which contains the custom `human_description` model).
EXAMPLE_ROBOT_DATA_MODEL_DIR = os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR", _VENDORED_ROBOTS)
EXAMPLE_ROBOT_DATA_SOURCE_DIR = os.environ.get("EXAMPLE_ROBOT_DATA_SOURCE_DIR", _VENDORED_ROBOTS)
