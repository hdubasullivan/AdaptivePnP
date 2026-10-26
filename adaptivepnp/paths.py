"""Default locations of the data and the model weights.

From a source checkout (``pip install -e .``) they are ``data/`` and ``model_zoo/`` under the repository root;
from an installed package, under the current working directory. Override either with ``ADAPTIVEPNP_DATA_DIR``
and ``ADAPTIVEPNP_MODEL_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    """The repository root when this file sits in a source checkout, else the current working directory."""
    checkout = Path(__file__).resolve().parent.parent
    return checkout if (checkout / "pyproject.toml").exists() else Path.cwd()


ROOT = _root()
DATA_DIR = Path(os.environ.get("ADAPTIVEPNP_DATA_DIR", ROOT / "data")).expanduser()
MODEL_ZOO = Path(os.environ.get("ADAPTIVEPNP_MODEL_DIR", ROOT / "model_zoo")).expanduser()
