#!/usr/bin/env python
"""Download the pretrained color DRUNet (KAIR release) into model_zoo/.

python scripts/download_weights.py [--model-dir model_zoo]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from adaptivepnp.denoisers import DRUNET_WEIGHTS_URL
from adaptivepnp.paths import MODEL_ZOO


def main() -> None:
    """Download the DRUNet weights into the model directory given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(MODEL_ZOO))
    args = parser.parse_args()
    target = Path(args.model_dir) / "drunet_color.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"{target} already exists")
        return
    print(f"downloading {DRUNET_WEIGHTS_URL} -> {target}")
    urllib.request.urlretrieve(DRUNET_WEIGHTS_URL, target)  # noqa: S310
    print(f"done ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
