# AdaptivePnP

Plug-and-play (PnP) image restoration that tunes itself. At every iteration the noise level `sigma` and the step
size `gamma` of the data-fitting step are estimated from the current residual `y - A v` by maximizing a profiled
Gaussian likelihood in the forward model's eigenbasis, with the search over `rho = gamma / sigma` restricted to an
*identifiable window*: the interval of ratios on which the normalized Cramér–Rao bounds of both `sigma²` and
`gamma²` are within a tolerance `tau`. Nothing is tuned by hand; the only inputs are the number of iterations `K`
and `tau`, which defaults to 0.05.

The package implements PnP-PGM and PnP-ADMM with a DRUNet denoiser for circular deblurring and for 2x
super-resolution (blur followed by decimation), with exact closed-form data-fitting steps. The demos compare
AdaptivePnP with the same PnP algorithm at a fixed, validation-tuned step size (TunedPnP).

## Quick start

```bash
git clone https://github.com/hdubasullivan/AdaptivePnP.git
cd AdaptivePnP
conda create -n adaptivepnp python=3.12 && conda activate adaptivepnp
pip install -e .
python scripts/download_weights.py      # model_zoo/drunet_color.pth (KAIR, ~130 MB)
python scripts/download_data.py         # data/kernels_12.mat and data/CBSD68
python demos/deblur.py           # AdaptivePnP vs TunedPnP on one image; outputs in demos/output/
```

The downloads go to `data/` and `model_zoo/` under the repository root, where runs look for them (with a
non-editable install, under the current working directory). If your data or weights live elsewhere, point the
environment variables `ADAPTIVEPNP_DATA_DIR` and `ADAPTIVEPNP_MODEL_DIR` at the corresponding directories.

## Demos

`demos/deblur.py` and `demos/super_resolution.py` take one CBSD68 image, blur it with one of the twelve benchmark
kernels of `kernels_12.mat` (isotropic Gaussian G1-G4, anisotropic Gaussian A1-A4, motion M1-M4; `--kernel 0` to
`11`), decimate it by 2 for super-resolution, add white noise, and reconstruct it from the back-projection `A^T y`
with two methods:

- **AdaptivePnP**, which estimates `(sigma, gamma)` at every iteration, and
- **TunedPnP**, the same PnP algorithm at a fixed step size `gamma = rho * sigma` with the true noise level, at
  the validation-tuned `rho` of our paper: 1.47 (PGM) and 2.15 (ADMM) for deblurring, 3.16 (PGM) and 4.64
  (ADMM) for 2x super-resolution. `--rho` sets another value; `--rho 0` skips the baseline.

Each demo prints the identifiable window, the per-iteration `(sigma_hat, gamma, rho)` and PSNR, the final PSNR
and SSIM of both methods, and the mean milliseconds per iteration (the whole iteration, one DRUNet evaluation and
the parameter estimate). Additionally, it writes outputs in `demos/output/<title>/`. It saves `results.json` 
(settings, identifiable window,  and per-method PSNR, SSIM, runtime, mean milliseconds per iteration and the 
per-iteration `sigma_hat`, `gamma`, `rho`, PSNR, SSIM and milliseconds), `iterations.csv` (the same per-iteration table), 
one full-resolution PNG per image  (ground truth, measurement, each reconstruction), `comparison.png` (the images with a 
zoomed crop and the metrics) and `convergence.png` (the per-iteration PSNR, SSIM, `sigma_hat`, `gamma` and `rho` of each method).

```bash
python demos/deblur.py --help                                           # image, kernel, noise, algorithm, K, tau, beta, device
python demos/deblur.py --kernel 8 --noise 0.05 --algorithm admm         # motion kernel M1, 5% noise, ADMM
python demos/super_resolution.py --kernel 2                             # 2x SR, kernel G3
```

The demos run on `cuda` or `mps` when available and on the CPU otherwise (about one second per iteration on a
laptop CPU); `--device` forces one.

## Using AdaptivePnP on your own problem

The operators work on any image whose spatial size exceeds the kernel (and, for super-resolution, is divisible
by the factor); `pnp_recon` runs the whole reconstruction and returns the result as a NumPy array.

```python
import torch
from adaptivepnp import Convolution, SuperResolution, make_initial_guess, pnp_recon
from adaptivepnp.denoisers import load_drunet
from adaptivepnp.device import select_device

device = select_device()                                  # cuda, mps or cpu
denoiser = load_drunet(device)                            # model_zoo/drunet_color.pth
kernel = ...                                              # 2-D blur kernel, a (kh, kw) float tensor
x = ...                                                   # (1, 3, H, W) float32 in [0, 1]; only to synthesize y
A = Convolution(kernel, input_shape=tuple(x.shape))       # or SuperResolution(kernel, 2, input_shape=tuple(x.shape))
y = (A(x) + 0.01 * torch.randn(A.output_shape)).to(device)

x0 = make_initial_guess(A, y)                             # A^T y
_, x_adaptive, iterates, curves = pnp_recon(y, x0, A, denoiser, "pgm", K=10, record_iterates=True)
_, x_tuned, _, _ = pnp_recon(y, x0, A, denoiser, "pgm", K=10, gamma=1.47 * 0.01, sigma=0.01)  # TunedPnP
```

`curves["sigma"]` and `curves["gamma"]` hold the per-iteration estimates and `iterates` every iterate; `"admm"`
selects PnP-ADMM and `tau=` the tolerance. For a real measurement build `A` with the `input_shape` of the image
you want to recover and pass your `y`; `Convolution` and `SuperResolution` are circular, so pad or taper the
borders of a non-periodic measurement as you would for any FFT-based method.

## Conventions

Three conventions of the demos matter when comparing their numbers with other papers' tables:

- CBSD68 is loaded from the HuggingFace dataset `deepinv/CBSD68`, so `--image` follows that dataset's order, not
  the file names of the original release.
- Landscape images are rotated to portrait on loading, so every image is 481 x 321 and the identifiable window,
  which depends only on the kernel and the image size, is the same for every image. Saved images are rotated back.
- PSNR and SSIM are computed on the unclipped RGB estimate after a 10-pixel border crop on every edge
  (`adaptivepnp/metrics.py`), with SSIM from scikit-image (7 x 7 uniform window, `channel_axis=-1`, data range 1).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no data and runs in a few seconds; it checks the operators, the closed-form proximal map, the
estimator and its window, the solvers and policies, and the metrics. A few end-to-end tests marked
`data` run the real DRUNet on one CBSD68 image; they run only when `data/` and `model_zoo/` are present and are
skipped otherwise.

## Troubleshooting

If a script aborts with `OMP: Error #15` (two OpenMP runtimes, common with conda PyTorch on macOS), set
`KMP_DUPLICATE_LIB_OK=TRUE` in the environment. If `import adaptivepnp` resolves to another checkout, run
with `PYTHONPATH=.` from the repository root or reinstall with `pip install -e .`.

## Citing

```bibtex
@article{dubasullivan2026adaptivepnp,
  title   = {AdaptivePnP: Fast Auto-Tuned Plug-and-Play for Deconvolution},
  author  = {Duba-Sullivan, Haley and Reid, Emma J. and Bouman, Charles A. and Buzzard, Gregery T.},
  year    = {2026},
  note    = {preprint}
}
```

## Attribution

The DRUNet denoiser (K. Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", IEEE TPAMI
44(10), 2022) in `adaptivepnp/denoisers.py` follows the reference implementation in
[KAIR](https://github.com/cszn/KAIR) (Kai Zhang, MIT License), and `scripts/download_weights.py` fetches
KAIR's released `drunet_color.pth`. The twelve blur kernels (`kernels_12.mat`) are also distributed by KAIR. The
MIT notice is reproduced in `THIRD_PARTY_LICENSES`.

## License

BSD 3-Clause. See `LICENSE`.
