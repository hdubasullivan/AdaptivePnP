"""Shared pieces of the two demos: argument parsing, the runs, the printed tables, and the saved results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from adaptivepnp.denoisers import load_drunet
from adaptivepnp.device import select_device
from adaptivepnp.estimator import RHO_SEARCH_BOUNDS, TAU_DEFAULT, identifiable_window_of
from adaptivepnp.metrics import psnr, ssim
from adaptivepnp.pnp import make_initial_guess, pnp_recon

OUT_DIR = Path(__file__).resolve().parent / "output"
#: The per-iteration milliseconds reported per method: the whole iteration, one DRUNet evaluation, the estimate.
TIMINGS = ("ms_iteration", "ms_denoise", "ms_ml_estimate")
CROP = 96  # side of the zoomed crop in the comparison figure, in pixels
#: The validation-tuned rho of the fixed-step baseline TunedPnP in the paper, per (problem, algorithm).
TUNED_RHO = {
    ("deblurring", "pgm"): 1.47,
    ("deblurring", "admm"): 2.15,
    ("super_resolution", "pgm"): 3.16,
    ("super_resolution", "admm"): 4.64,
}


def parser(description: str) -> argparse.ArgumentParser:
    """Parse the common arguments of both demos.

    Args:
        description: One-line description shown by ``--help``.

    Returns:
        Parser.

    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--image", type=int, default=0, help="CBSD68 image index 0-67")
    p.add_argument(
        "--kernel", type=int, default=8, help="kernel index in kernels_12.mat: G1-4 = 0-3, A1-4 = 4-7, M1-4 = 8-11"
    )
    p.add_argument("--noise", type=float, default=0.01, help="measurement noise std on [0, 1] images")
    p.add_argument("--K", type=int, default=10, help="PnP iterations")
    p.add_argument("--algorithm", choices=("pgm", "admm"), default="pgm")
    p.add_argument("--tau", type=float, default=TAU_DEFAULT, help="identifiability tolerance (Theorem 1)")
    p.add_argument("--beta", type=float, default=1.0, help="EMA weight on the step-size estimate (1 = none)")
    p.add_argument(
        "--rho",
        type=float,
        default=None,
        help="rho of the fixed-step baseline TunedPnP, gamma = rho * noise (default: the paper's tuned value for this "
        "problem and algorithm; 0 skips the baseline)",
    )
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    return p


def run(args, x: torch.Tensor, y: torch.Tensor, operator, title: str, *, rotated: bool = False) -> dict:
    """Reconstruct with AdaptivePnP and TunedPnP, print the results, and save them.

    Args:
        args: Parsed demo arguments.
        x: Ground-truth image ``(1, 3, H, W)``.
        y: Measurement.
        operator: Forward operator that produced ``y``.
        title: Title of the printed block and of the saved figure.
        rotated: Whether ``x`` was rotated to portrait on loading; saved images are rotated back.

    Returns:
        The results dictionary that ``--save`` writes to ``results.json``.

    """
    device = args.device or select_device()
    denoiser = load_drunet(device)
    y_dev = y.to(device)
    x0 = make_initial_guess(operator, y_dev, "adjoint").clone()

    problem = "super_resolution" if getattr(operator, "sampling", None) is not None else "deblurring"
    rho = TUNED_RHO[(problem, args.algorithm)] if args.rho is None else args.rho
    lo, hi = identifiable_window_of(operator, args.tau)
    print(f"{title}: device {device}, K = {args.K}, {args.algorithm.upper()}")
    print(f"identifiable window at tau = {args.tau}: rho in [{lo:.3g}, {hi:.3g}]")

    results = {
        "title": title,
        "settings": {**{k: v for k, v in vars(args).items() if k != "save"}, "rho": rho},
        "device": device,
        "identifiable_window": {"rho_inf": lo, "rho_sup": hi},
        "methods": {},
    }
    panels = {"Ground truth": (x, None), "Measurement": (_measurement_panel(y, operator), None)}

    # AdaptivePnP
    t0 = time.perf_counter()
    _, recon, iterates, curves = pnp_recon(
        y_dev, x0, operator, denoiser, args.algorithm, K=args.K, beta=args.beta, tau=args.tau, record_iterates=True
    )
    elapsed = time.perf_counter() - t0
    x_adaptive = torch.from_numpy(recon)
    rows = [
        {"k": k, "sigma_hat": s, "gamma": g, "rho": g / s, "psnr": psnr(it, x), "ssim": ssim(it, x)}
        for k, (s, g, it) in enumerate(zip(curves["sigma"], curves["gamma"], iterates))
    ]
    results["methods"]["AdaptivePnP"] = _summary(x_adaptive, x, elapsed, rows, curves)
    panels["AdaptivePnP"] = (x_adaptive, results["methods"]["AdaptivePnP"])

    print("\nAdaptivePnP (sigma and gamma estimated every iteration):")
    print(f"  {'k':>2s} {'sigma_hat':>10s} {'gamma':>10s} {'rho':>9s} {'PSNR':>7s}")
    for r in rows:
        print(f"  {r['k']:2d} {r['sigma_hat']:10.5f} {r['gamma']:10.5f} {r['rho']:9.3f} {r['psnr']:7.2f}")
    print(f"  PSNR {rows[-1]['psnr']:.2f} dB, SSIM {rows[-1]['ssim']:.3f}  (true sigma {args.noise}), {elapsed:.1f} s")
    print(f"  {_timing_line(results['methods']['AdaptivePnP'])}")
    at_edge = [r["k"] for r in rows if r["rho"] <= lo * 1.001 or r["rho"] >= hi * 0.999]
    if at_edge:
        edge = "lower edge rho_inf" if rows[at_edge[0]]["rho"] <= lo * 1.001 else "upper edge rho_sup"
        contiguous = at_edge == list(range(at_edge[0], len(rows)))
        when = f"starting at iteration {at_edge[0]}" if contiguous else f"at iteration(s) {at_edge}"
        print(f"  rho_k sits at the window's {edge} {when}.")

    # TunedPnP: fixed-step PnP with the true noise level
    if rho > 0:
        t0 = time.perf_counter()
        _, recon, iterates, curves = pnp_recon(
            y_dev,
            x0,
            operator,
            denoiser,
            args.algorithm,
            K=args.K,
            gamma=rho * args.noise,
            sigma=args.noise,
            record_iterates=True,
        )
        elapsed = time.perf_counter() - t0
        x_fixed = torch.from_numpy(recon)
        rows = [
            {
                "k": k,
                "sigma_hat": args.noise,
                "gamma": rho * args.noise,
                "rho": rho,
                "psnr": psnr(it, x),
                "ssim": ssim(it, x),
            }
            for k, it in enumerate(iterates)
        ]
        name = f"TunedPnP (rho = {rho:g})" if args.rho is None else f"PnP (rho = {rho:g})"
        results["methods"][name] = _summary(x_fixed, x, elapsed, rows, curves)
        panels[name] = (x_fixed, results["methods"][name])
        print(
            f"\n{name} with the true sigma: PSNR {rows[-1]['psnr']:.2f} dB, SSIM {rows[-1]['ssim']:.3f}, {elapsed:.1f} s"
        )
        print(f"  {_timing_line(results['methods'][name])}")

    _save(results, panels, title, rotated)
    return results


def _summary(x_hat: torch.Tensor, x: torch.Tensor, elapsed: float, rows: list[dict], curves: dict) -> dict:
    """The per-method entry of ``results.json``.

    Besides PSNR, SSIM and the run's seconds it holds the mean milliseconds per iteration (over iterations 1
    to K) of the whole iteration, of one DRUNet evaluation and of the parameter estimate, and the per-iteration
    rows, each extended with the same milliseconds for its own iterate (NaN at iterate 0, which ran none).
    """
    for r in rows:
        r.update({key: curves[key][r["k"]] for key in TIMINGS})
    timing = {key: float(np.nanmean(curves[key][1:])) for key in TIMINGS}
    return {"psnr": psnr(x_hat, x), "ssim": ssim(x_hat, x), "time_s": elapsed, **timing, "iterations": rows}


def _timing_line(entry: dict) -> str:
    """A method's mean milliseconds per iteration, as one printed line."""
    return (
        f"per iteration {entry['ms_iteration']:.1f} ms: DRUNet {entry['ms_denoise']:.1f} ms, "
        f"parameter estimate {entry['ms_ml_estimate']:.2f} ms"
    )


def _measurement_panel(y: torch.Tensor, operator) -> torch.Tensor:
    """The measurement on the image grid: pixel-replicated for super-resolution."""
    sampling = getattr(operator, "sampling", None)
    if sampling is None:
        return y
    fy, fx = sampling.factor
    return y.repeat_interleave(fy, dim=-2).repeat_interleave(fx, dim=-1)


def _to_image(t: torch.Tensor, rotated: bool) -> np.ndarray:
    """A ``(1, 3, H, W)`` tensor as an ``(H, W, 3)`` array in ``[0, 1]``, rotated back to its original orientation."""
    t = t.detach().float().cpu()
    if rotated:
        t = torch.rot90(t, -1, dims=(-2, -1))
    return t[0].clamp(0, 1).permute(1, 2, 0).numpy()


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower().replace("%", "pct")).strip("_")


def _save(results: dict, panels: dict, title: str, rotated: bool) -> None:
    """Write ``results.json``, ``iterations.csv``, one PNG per panel and the comparison figure to ``demos/output/<slug>/``."""
    from PIL import Image

    folder = OUT_DIR / _slug(title)
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (folder / "iterations.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "k", "sigma_hat", "gamma", "rho", "psnr", "ssim", *TIMINGS])
        for name, entry in results["methods"].items():
            for r in entry["iterations"]:
                writer.writerow(
                    [name, r["k"], r["sigma_hat"], r["gamma"], r["rho"], r["psnr"], r["ssim"]]
                    + [r[key] for key in TIMINGS]
                )

    images = {name: _to_image(tensor, rotated) for name, (tensor, _) in panels.items()}
    for name, image in images.items():
        Image.fromarray((image * 255).round().astype(np.uint8)).save(folder / f"{_slug(name)}.png")

    _comparison_figure(images, panels, title, folder / "comparison.png")
    _convergence_figure(results, title, folder / "convergence.png")
    print(
        f"\nsaved results.json, iterations.csv, {len(images)} image(s), comparison.png and convergence.png to {folder}"
    )


def _comparison_figure(images: dict, panels: dict, title: str, path: Path) -> None:
    """Full images on top, a zoomed crop below, metrics in the titles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = list(images)
    h, w = next(iter(images.values())).shape[:2]
    top, left = (h - CROP) // 2, (w - CROP) // 2
    fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 2.6 * (h / w + 1.0)), facecolor="white")
    for j, name in enumerate(names):
        image = images[name]
        summary = panels[name][1]
        label = name if summary is None else f"{name}\n{summary['psnr']:.2f} dB / {summary['ssim']:.3f}"
        axes[0, j].imshow(image)
        axes[0, j].set_title(label, fontsize=9)
        axes[0, j].add_patch(Rectangle((left, top), CROP, CROP, fill=False, edgecolor="yellow", linewidth=1))
        axes[1, j].imshow(image[top : top + CROP, left : left + CROP], interpolation="nearest")
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _convergence_figure(results: dict, title: str, path: Path) -> None:
    """PSNR, SSIM, ``sigma_hat``, ``gamma`` and ``rho`` against the iteration, one line per method.

    The true noise level and the identifiable window's edges are drawn as reference lines.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    quantities = (
        ("psnr", "PSNR (dB)", "linear"),
        ("ssim", "SSIM", "linear"),
        ("sigma_hat", "sigma", "linear"),
        ("gamma", "gamma", "log"),
        ("rho", "rho = gamma / sigma", "log"),
    )
    fig, axes = plt.subplots(1, len(quantities), figsize=(3.2 * len(quantities), 3.0), facecolor="white")
    for ax, (key, label, scale) in zip(axes, quantities):
        if key == "sigma_hat":
            ax.axhline(results["settings"]["noise"], color="black", linestyle="--", linewidth=1, label="true sigma")
        if key == "rho":
            window = results["identifiable_window"]
            ax.axhline(window["rho_inf"], color="black", linestyle=":", linewidth=1, label="rho_inf")
            if window["rho_sup"] < RHO_SEARCH_BOUNDS[1]:
                ax.axhline(window["rho_sup"], color="black", linestyle="-.", linewidth=1, label="rho_sup")
        for name, entry in results["methods"].items():
            ks = [r["k"] for r in entry["iterations"]]
            ax.plot(ks, [r[key] for r in entry["iterations"]], marker="o", markersize=3, label=name)
        ax.set_yscale(scale)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("iteration k")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
