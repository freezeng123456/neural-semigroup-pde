#!/usr/bin/env python3
"""Checkpoint-only common-path attribution for the Fisher learned-tube screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate_fisher_generator_consistency import (  # noqa: E402
    load_latent_checkpoint,
    sha256_file,
)
from experiment_artifacts import torch_load_compat  # noqa: E402
from fisher_generator_metrics import (  # noqa: E402
    fisher_spectral_generator,
    generator_residual_metrics,
    learned_generator_tube_states,
    learned_physical_generator,
    stability_pair_metrics,
    trapezoid_time_weights,
)


TIMES = (0.0, 0.3, 0.6, 0.9, 1.2)
PAIR_FAMILY = "sinusoidal_0.01"
CONDITIONING_TIME = 0.075


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().reshape(-1).to(torch.float64).cpu()
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("attribution values contain a non-finite entry")
    return {
        "mean": float(values.mean().item()),
        "rms": float(torch.sqrt(values.square().mean()).item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.9).item()),
        "max": float(values.max().item()),
    }


@torch.no_grad()
def directional_defect_metrics(
    model,
    states: torch.Tensor,
    reference_states: torch.Tensor,
    *,
    length: float,
    diffusivity: float,
    reaction_rate: float,
    batch_size: int,
) -> dict[str, object]:
    """Measure the defect component that injects current trajectory error."""

    signed_rates = []
    harmful_components = []
    cosines = []
    error_norms = []
    dx = float(length) / int(states.shape[-1])
    for start in range(0, len(states), int(batch_size)):
        batch = states[start : start + int(batch_size)]
        target = reference_states[start : start + int(batch_size)].to(batch.device)
        learned, represented, _latent = learned_physical_generator(
            model,
            batch,
            conditioning_time=CONDITIONING_TIME,
        )
        reference_rhs = fisher_spectral_generator(
            represented,
            length=length,
            diffusivity=diffusivity,
            reaction_rate=reaction_rate,
        )
        defect = learned - reference_rhs
        error = represented - target
        error_norm = torch.sqrt(dx * error.square().sum(dim=1))
        defect_norm = torch.sqrt(dx * defect.square().sum(dim=1))
        inner = dx * (error * defect).sum(dim=1)
        denominator = error_norm * defect_norm
        cosine = torch.where(
            denominator > 1e-14,
            inner / denominator.clamp_min(1e-14),
            torch.zeros_like(inner),
        )
        rate = torch.where(
            error_norm.square() > 1e-14,
            inner / error_norm.square().clamp_min(1e-14),
            torch.zeros_like(inner),
        )
        harmful = torch.where(
            error_norm > 1e-14,
            torch.relu(inner) / error_norm.clamp_min(1e-14),
            torch.zeros_like(inner),
        )
        signed_rates.append(rate.cpu())
        harmful_components.append(harmful.cpu())
        cosines.append(cosine.cpu())
        error_norms.append(error_norm.cpu())
    return {
        "signed_error_energy_rate": distribution(torch.cat(signed_rates)),
        "harmful_defect_component_l2": distribution(
            torch.cat(harmful_components)
        ),
        "error_defect_cosine": distribution(torch.cat(cosines)),
        "trajectory_error_l2": distribution(torch.cat(error_norms)),
    }


def reference_tube(trajectories, reference_dt: float) -> torch.Tensor:
    indices = [int(round(time / reference_dt)) for time in TIMES]
    return torch.stack(
        [
            torch.stack([states[index] for _times, states in trajectories])
            for index in indices
        ],
        dim=1,
    )


def weighted_rms(per_time: dict[str, dict[str, object]], key: str) -> float:
    weights = trapezoid_time_weights(TIMES).tolist()
    return math.sqrt(
        sum(
            float(weight) * float(per_time[str(time)][key]) ** 2
            for time, weight in zip(TIMES, weights)
        )
    )


def evaluate_path(
    states: torch.Tensor,
    reference: torch.Tensor,
    models: dict[str, torch.nn.Module],
    *,
    cache_config: dict[str, object],
    batch_size: int,
) -> dict[str, object]:
    result: dict[str, object] = {"times": list(TIMES), "models": {}}
    length = float(cache_config["L"])
    diffusivity = float(cache_config["nu"])
    reaction_rate = float(cache_config["reaction_rate"])
    for model_name, model in models.items():
        per_time: dict[str, dict[str, object]] = {}
        for index, time_value in enumerate(TIMES):
            current = states[:, index].to(next(model.parameters()).device)
            target = reference[:, index].to(current.device)
            generator = generator_residual_metrics(
                model,
                current,
                conditioning_time=CONDITIONING_TIME,
                length=length,
                diffusivity=diffusivity,
                reaction_rate=reaction_rate,
                batch_size=batch_size,
            )
            stability = stability_pair_metrics(
                model,
                current,
                pair_family=PAIR_FAMILY,
                conditioning_time=CONDITIONING_TIME,
                length=length,
                diffusivity=diffusivity,
                reaction_rate=reaction_rate,
                batch_size=batch_size,
            )
            directional = directional_defect_metrics(
                model,
                current,
                target,
                length=length,
                diffusivity=diffusivity,
                reaction_rate=reaction_rate,
                batch_size=batch_size,
            )
            per_time[str(time_value)] = {
                "generator": generator,
                "stability": stability,
                "directional_defect": directional,
            }
        result["models"][model_name] = {"per_time": per_time}
    return result


def path_comparison(path: dict[str, object]) -> dict[str, float]:
    models = path["models"]
    baseline = models["baseline"]["per_time"]
    tube = models["tube"]["per_time"]
    baseline_residual = weighted_rms(
        {time: value["generator"] for time, value in baseline.items()},
        "rms_residual_l2",
    )
    tube_residual = weighted_rms(
        {time: value["generator"] for time, value in tube.items()},
        "rms_residual_l2",
    )
    weights = trapezoid_time_weights(TIMES).tolist()
    baseline_harmful = math.sqrt(
        sum(
            weight
            * baseline[str(time)]["directional_defect"][
                "harmful_defect_component_l2"
            ]["rms"]
            ** 2
            for time, weight in zip(TIMES, weights)
        )
    )
    tube_harmful = math.sqrt(
        sum(
            weight
            * tube[str(time)]["directional_defect"][
                "harmful_defect_component_l2"
            ]["rms"]
            ** 2
            for time, weight in zip(TIMES, weights)
        )
    )
    baseline_osl = sum(
        weight
        * baseline[str(time)]["stability"]["learned_quotient"]["p95"]
        for time, weight in zip(TIMES, weights)
    )
    tube_osl = sum(
        weight * tube[str(time)]["stability"]["learned_quotient"]["p95"]
        for time, weight in zip(TIMES, weights)
    )
    return {
        "baseline_weighted_generator_rms": baseline_residual,
        "tube_weighted_generator_rms": tube_residual,
        "tube_over_baseline_generator_rms": tube_residual / baseline_residual,
        "baseline_weighted_harmful_component_rms": baseline_harmful,
        "tube_weighted_harmful_component_rms": tube_harmful,
        "tube_minus_baseline_weighted_osl_p95": tube_osl - baseline_osl,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--tube-checkpoint", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    cache = torch_load_compat(args.test_cache, map_location="cpu")
    initial = cache["test_u0"][: args.n_test]
    trajectories = cache["test_trajs"][: args.n_test]
    cache_config = cache["locked_test_config"]
    baseline, _baseline_metadata = load_latent_checkpoint(
        args.baseline_checkpoint, device
    )
    tube, _tube_metadata = load_latent_checkpoint(args.tube_checkpoint, device)
    models = {"baseline": baseline, "tube": tube}
    reference = reference_tube(
        trajectories, float(cache_config["reference_dt"])
    )
    paths = {
        "reference": reference,
        "baseline_learned": learned_generator_tube_states(
            baseline, initial.to(device), TIMES
        ).cpu(),
        "tube_learned": learned_generator_tube_states(
            tube, initial.to(device), TIMES
        ).cpu(),
    }
    evaluations = {
        name: evaluate_path(
            states,
            reference,
            models,
            cache_config=cache_config,
            batch_size=args.batch_size,
        )
        for name, states in paths.items()
    }
    result = {
        "experiment": "fisher_tube_common_path_attribution",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": int(args.seed),
        "n_test": int(args.n_test),
        "times": list(TIMES),
        "pair_family": PAIR_FAMILY,
        "inputs_sha256": {
            "baseline_checkpoint": sha256_file(args.baseline_checkpoint),
            "tube_checkpoint": sha256_file(args.tube_checkpoint),
            "test_cache": sha256_file(args.test_cache),
        },
        "paths": evaluations,
        "comparison": {
            name: path_comparison(value) for name, value in evaluations.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
