"""Super-resolve one CBSD68 image with AdaptivePnP.

    python demos/super_resolution.py                 # kernel G4, 1% noise, PGM, K = 10: AdaptivePnP vs TunedPnP
    python demos/super_resolution.py --kernel 9 --noise 0.03 --algorithm admm --save

Needs data/kernels_12.mat and data/CBSD68 (scripts/download_data.py) and model_zoo/drunet_color.pth
(scripts/download_weights.py). The fixed-step baseline TunedPnP runs with the true noise level at the paper's
validation-tuned rho for 2x super-resolution, 3.16 (PGM) or 4.64 (ADMM); ``--rho`` sets another value and
``--rho 0`` skips it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import parser, run

from adaptivepnp.data import load_cbsd68, super_resolution_problem
from adaptivepnp.kernels import kernel_label, load_kernels


def main() -> None:
    """Parse the arguments, build the super-resolution problem, and run the demo."""
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--factor", type=int, default=2, help="decimation factor")
    args = p.parse_args()

    x, rotated = load_cbsd68(args.image, return_rotation=True)
    kernel = load_kernels()[args.kernel]
    x, y, operator = super_resolution_problem(x, kernel, args.factor, args.noise)
    run(
        args,
        x,
        y,
        operator,
        f"{args.factor}x super-resolution, kernel {kernel_label(args.kernel)}, {100 * args.noise:g}% noise",
        rotated=rotated,
    )


if __name__ == "__main__":
    main()
