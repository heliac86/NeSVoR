#!/usr/bin/env python3
"""Diagnose FRAME patch coordinates in the installed NeSVoR environment.

Usage: python check_ff_patch_coordinates.py --input /path/to/degraded.nii

No installed source files are edited. The installed console entry point loads
one target. A process-local hook runs after intensity normalization, captures
the real patch_forward coordinates before the density network, then exits before
optimizer construction. No reconstruction/model output is saved by this script.

Exit codes: 0 = sampled coordinates agree; 2 = coordinate mismatch detected;
3 = inconclusive reference match; 1 = environment/hook error.
"""

import argparse
import ast
import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import textwrap
import traceback
import types
from importlib import metadata

SCRIPT_VERSION = "3"


class _DiagnosticFinished(BaseException):
    def __init__(self, status):
        self.status = status


class _PatchCaptured(BaseException):
    pass


def _console_entry():
    entries = metadata.entry_points()
    if hasattr(entries, "select"):
        matches = list(entries.select(group="console_scripts", name="nesvor"))
    else:
        matches = [e for e in entries.get("console_scripts", []) if e.name == "nesvor"]
    if len(matches) != 1:
        raise RuntimeError("Expected one installed 'nesvor' console entry. Activate its environment.")
    return matches[0]


def _find_training_components(train_module="nesvor.inr.train"):
    # Console entry points may import reconstruction/training lazily.
    # Explicitly import training before discovering and installing the hook.
    try:
        loaded = importlib.import_module(train_module)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import training module {train_module!r}: "
            f"{type(exc).__name__}: {exc}. No training started."
        ) from exc
    print("Explicitly imported train module:", train_module, flush=True)
    print("Train module file:", getattr(loaded, "__file__", "<unknown>"), flush=True)
    candidates = {}
    observations = []
    for name, module in list(sys.modules.items()):
        if not (name == "nesvor" or name.startswith("nesvor.")) or module is None:
            continue
        fn = vars(module).get("train")
        if not isinstance(fn, types.FunctionType):
            continue
        glob = fn.__globals__
        if "PointDataset" not in glob or "NeSVoR" not in glob:
            continue
        cls = glob["PointDataset"]
        path = inspect.getsourcefile(fn)
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            calls_normalize = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "normalize"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "dataset"
                for node in ast.walk(tree)
            )
        except (OSError, TypeError, SyntaxError):
            calls_normalize = False
        has_normalize = callable(getattr(cls, "normalize", None))
        observations.append(
            f"{name}.train: file={path}; "
            f"dataset.normalize method={has_normalize}; "
            f"train calls dataset.normalize={calls_normalize}"
        )
        if calls_normalize and has_normalize:
            candidates[id(fn)] = (fn, cls, glob["NeSVoR"])
    if len(candidates) != 1:
        for item in dict.fromkeys(observations):
            print("Training discovery:", item, flush=True)
        raise RuntimeError(
            f"Found {len(candidates)} compatible train() functions after explicit import. "
            "The installed source may differ from the supplied normalized-target version. "
            "See the module paths and discovery details above. No training started."
        )
    return next(iter(candidates.values()))


def _coordinate_probe(dataset, model, point_count, verbose):
    import torch

    model_module = importlib.import_module(type(model).__module__)
    transform = getattr(model_module, "ax_transform_points")
    scale = float(model.spatial_scaling)
    if scale <= 0:
        raise RuntimeError("Invalid spatial_scaling.")
    if getattr(model.args, "deformable", False):
        raise RuntimeError("This diagnostic expects the final non-deformable configuration.")

    with torch.no_grad():
        batch = dataset.get_patch_batch(
            n_patches=8, patch_size=48, device=dataset.xyz.device
        )
        if not batch:
            raise RuntimeError("No valid patch found. Try another target or seed.")
        # One real patch is enough; retain the entire patch for patch_forward.
        batch = {key: value[:1] for key, value in batch.items()}
        slice_id = int(batch["slice_idx_patch"][0].item())
        valid_ids = batch["valid_mask_patch"][0].reshape(-1).nonzero().flatten()
        if not valid_ids.numel():
            raise RuntimeError("Selected patch has no valid pixels.")
        raw = batch["xyz_patch"][0]
        order = raw[valid_ids].norm(dim=-1).argsort()
        count = min(point_count, int(valid_ids.numel()))
        ranks = torch.linspace(0, int(valid_ids.numel()) - 1, count,
                               device=valid_ids.device).round().long()
        chosen = valid_ids[order[ranks]].unique()
        raw_chosen = raw[chosen]
        proposed_local = raw_chosen / scale

        reference_mask = dataset.slice_idx == slice_id
        reference_xyz = dataset.xyz[reference_mask]
        reference_v = dataset.v[reference_mask]
        if not reference_xyz.numel():
            raise RuntimeError("No main-path pixels found for the selected slice.")
        matched = torch.stack([
            (reference_xyz - q).square().sum(-1).argmin()
            for q in proposed_local
        ])
        main_local = reference_xyz[matched]
        observed_patch_v = batch["v_patch"][0].reshape(-1)[chosen]
        local_error = (main_local - proposed_local).norm(dim=-1) * scale
        intensity_error = (reference_v[matched] - observed_patch_v).abs()

        # Intercept the actual patch path, including its actual PSF noise.
        captured = {}
        noises = []
        old_randn = torch.randn

        def record_randn(*args, **kwargs):
            result = old_randn(*args, **kwargs)
            noises.append(result)
            return result

        def capture_forward(xyz, se=None):
            captured["xyz"] = xyz
            raise _PatchCaptured()

        sentinel = object()
        previous = {name: model.__dict__.get(name, sentinel)
                    for name in ("net_forward", "net_forward_ff_direct")}
        try:
            torch.randn = record_randn
            model.net_forward = capture_forward
            model.net_forward_ff_direct = capture_forward
            try:
                model.patch_forward(**batch)
            except _PatchCaptured:
                pass
        finally:
            torch.randn = old_randn
            for name, value in previous.items():
                if value is sentinel:
                    delattr(model, name)
                else:
                    setattr(model, name, value)

        if "xyz" not in captured or len(noises) != 1:
            raise RuntimeError("Unexpected patch_forward structure; capture cannot be interpreted.")
        noise = noises[0][chosen]
        actual = captured["xyz"][chosen]
        if noise.shape != actual.shape:
            raise RuntimeError("Captured PSF noise and coordinates have incompatible shapes.")
        if verbose:
            print("Captured installed patch path; comparing with main-path coordinates.", flush=True)
        # expand() creates a zero-stride view. NeSVoR's CUDA axis-angle
        # converter requires contiguous storage, unlike ordinary torch ops.
        # This copy changes storage layout only, not transformation values.
        t = (model.axisangle[slice_id].reshape(1, 1, -1)
             .expand(len(chosen), 1, -1).contiguous())
        sigma = model.psf_sigma[slice_id].view(1, 1, 3)
        # Same noise, same transformation, different local coordinate sources.
        expected_main = transform(t, (main_local[:, None] + noise * sigma).contiguous(),
                                  model.trans_first)
        proposed = transform(t, (proposed_local[:, None] + noise * sigma).contiguous(),
                             model.trans_first)
        actual_error = (actual - expected_main).norm(dim=-1).amax(dim=1) * scale
        proposed_error = (proposed - expected_main).norm(dim=-1).amax(dim=1) * scale

        bb = model.inr.bounding_box
        norm_main = (expected_main.mean(1) - bb[0]) / (bb[1] - bb[0])
        norm_actual = (actual.mean(1) - bb[0]) / (bb[1] - bb[0])
        max_local = float(local_error.max().item())
        max_intensity = float(intensity_error.max().item())
        max_actual = float(actual_error.max().item())
        max_proposed = float(proposed_error.max().item())
        print("\n=== FRAME coordinate diagnostic ===", flush=True)
        print(f"slice_index: {slice_id}; tested_pixels: {len(chosen)}; PSF_samples: {noise.shape[1]}")
        print(f"spatial_scaling: {scale:g}")
        print(f"reference_local_match_max_mm: {max_local:.9g}")
        print(f"observed_intensity_difference_max: {max_intensity:.9g}")
        print(f"actual_patch_vs_main_max_mm: {max_actual:.9g}")
        print(f"proposed_divide_by_scale_vs_main_max_mm: {max_proposed:.9g}")
        print("Intensity errors use normalized intensities; spatial errors use physical mm.")
        worst = int(actual_error.argmax().item())
        print("Worst sampled point, mean normalized coordinates:")
        print("  main :", norm_main[worst].cpu().tolist())
        print("  patch:", norm_actual[worst].cpu().tolist())
        if verbose:
            for i in range(len(chosen)):
                print(f"pixel_flat_index={int(chosen[i].item())}: "
                      f"actual_error_mm={float(actual_error[i].item()):.9g}, "
                      f"proposed_error_mm={float(proposed_error[i].item()):.9g}")
        reference_ok = max_local <= 1e-3 and torch.allclose(
            reference_v[matched], observed_patch_v, atol=1e-6, rtol=1e-5
        )
        if not reference_ok or max_proposed > 1e-3:
            print("RESULT: INCONCLUSIVE. Check pixel identity, coordinate construction, and intensity normalization.")
            return 3
        if max_actual > 1e-3:
            print("RESULT: MISMATCH. The installed patch path differs from the main path; the /scale candidate agrees.")
            return 2
        print("RESULT: MATCH. The sampled installed patch coordinates agree with the main path.")
        return 0


def _run_installed(cli_args, point_count, verbose, train_module="nesvor.inr.train"):
    old_argv = sys.argv[:]
    original_normalize = None
    dataset_class = None
    try:
        sys.argv = ["nesvor"] + cli_args
        entry = _console_entry()
        print("Diagnostic script version:", SCRIPT_VERSION, flush=True)
        print("Python:", sys.executable, flush=True)
        print("Console entry:", entry.value, flush=True)
        package = importlib.import_module("nesvor")
        print("Installed package:", getattr(package, "__file__", "<namespace package>"), flush=True)
        console_main = entry.load()
        train_fn, dataset_class, model_class = _find_training_components(train_module)
        print("Installed train:", inspect.getsourcefile(train_fn), flush=True)
        print("Installed data:", inspect.getsourcefile(dataset_class), flush=True)
        print("Installed model:", inspect.getsourcefile(model_class), flush=True)
        print("Diagnostic mode: stop after normalization, before optimizer construction.", flush=True)
        original_normalize = dataset_class.normalize

        def normalize_then_probe(self, value):
            frame = inspect.currentframe().f_back
            try:
                context = frame.f_locals
                model = context.get("model")
                if context.get("dataset") is not self or not isinstance(model, model_class):
                    raise RuntimeError("normalize() was called outside the expected train context.")
                original_normalize(self, value)
                status = _coordinate_probe(self, model, point_count, verbose)
            finally:
                del frame
            raise _DiagnosticFinished(status)

        dataset_class.normalize = normalize_then_probe
        try:
            console_main()
        except _DiagnosticFinished as done:
            return done.status
        raise RuntimeError("The diagnostic hook was not reached.")
    finally:
        if original_normalize is not None:
            dataset_class.normalize = original_normalize
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One degraded NIfTI stack, as used by run_G14s_GA.sh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="NeSVoR device ID; default: 0")
    parser.add_argument("--points", type=int, default=5, help="Valid pixels to test from one real patch")
    parser.add_argument("--verbose", action="store_true", help="Print individual pixel errors")
    parser.add_argument("--train-module", default="nesvor.inr.train",
                        help="Training module to import explicitly; default: nesvor.inr.train")
    opts = parser.parse_args()
    input_path = Path(opts.input).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if opts.points < 1:
        parser.error("--points must be positive")
    with tempfile.TemporaryDirectory(prefix="frame-coordinate-check-") as tmp:
        cli_args = [
            "reconstruct", "--input-stacks", str(input_path),
            "--output-volume", str(Path(tmp) / "unused_diagnostic_output.nii.gz"),
            "--sample-mask", str(input_path), "--output-intensity-mean", "308",
            "--output-resolution", "1.0", "--no-transformation-optimization",
            "--registration", "none", "--single-precision",
            "--patch-size", "48", "--n-patches", "8", "--weight-image", "1.0",
            "--delta", "0.05", "--n-iter", "2000", "--batch-size", "4096",
            "--learning-rate", "5e-3", "--gamma", "0.33",
            "--milestones", "0.5", "0.75", "0.9", "--seed", str(opts.seed),
            "--device", str(opts.device), "--weight-ff-loss", "0.05", "--ff-alpha", "2.0",
            "--spatial-gating", "--gating-hidden-dim", "8", "--gating-activation", "silu",
            "--gating-grad-clip", "0.1", "--gating-lr-scale", "2.0",
        ]
        return _run_installed(cli_args, opts.points, opts.verbose, opts.train_module)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"DIAGNOSTIC ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "--verbose" in sys.argv:
            traceback.print_exc()
        sys.exit(1)
