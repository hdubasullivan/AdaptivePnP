#!/usr/bin/env python
"""Fetch the benchmark kernels and the CBSD68 test images into data/.

    python scripts/download_data.py [--data-dir data]

* ``kernels_12.mat`` -- the 12 blur kernels of KAIR's benchmark.
* ``CBSD68/``       -- the 68 test images, from the HuggingFace dataset ``deepinv/CBSD68`` saved to disk.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from adaptivepnp.paths import DATA_DIR

KERNELS_URL = "https://raw.githubusercontent.com/cszn/KAIR/master/kernels/kernels_12.mat"


def main() -> None:
    """Download the kernels and CBSD68 into the data directory given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    kernels = data_dir / "kernels_12.mat"
    if kernels.exists():
        print(f"{kernels} already exists")
    else:
        print(f"downloading {KERNELS_URL} -> {kernels}")
        urllib.request.urlretrieve(KERNELS_URL, kernels)  # noqa: S310

    cbsd = data_dir / "CBSD68"
    if cbsd.exists():
        print(f"{cbsd} already exists")
    else:
        from datasets import load_dataset

        print("downloading deepinv/CBSD68 from the HuggingFace hub")
        ds = load_dataset("deepinv/CBSD68", split="train")
        ds.save_to_disk(str(cbsd))
        print(f"saved {len(ds)} images to {cbsd}")


if __name__ == "__main__":
    main()
