"""Deblur one CBSD68 image with AdaptivePnP.

    python demos/deblur.py                          # kernel G4, 1% noise, PGM, K = 10: AdaptivePnP vs TunedPnP
    python demos/deblur.py --kernel 8 --noise 0.05 --algorithm admm --save

Needs data/kernels_12.mat and data/CBSD68 (scripts/download_data.py) and model_zoo/drunet_color.pth
(scripts/download_weights.py). The fixed-step baseline TunedPnP runs with the true noise level at the paper's
validation-tuned rho for deblurring, 1.47 (PGM) or 2.15 (ADMM); ``--rho`` sets another value and ``--rho 0`` skips
it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import parser, run

from adaptivepnp.data import deblurring_problem, load_cbsd68
from adaptivepnp.kernels import kernel_label, load_kernels


def main() -> None:
    """Parse the arguments, build the deblurring problem, and run the demo."""
    p = parser(__doc__.splitlines()[0])
    args = p.parse_args()

    x, rotated = load_cbsd68(args.image, return_rotation=True)
    kernel = load_kernels()[args.kernel]
    y, operator = deblurring_problem(x, kernel, args.noise)
    run(
        args,
        x,
        y,
        operator,
        f"Deblurring, kernel {kernel_label(args.kernel)}, {100 * args.noise:g}% noise",
        rotated=rotated,
    )


if __name__ == "__main__":
    main()
