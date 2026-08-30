"""Physical-generator and stability metrics for frozen Fisher checkpoints.

The historical Fisher models evolve a latent variable ``z`` and decode it by
``u = sigmoid(z)``.  This module exposes the corresponding physical-coordinate
vector field, the periodic spectral Fisher method-of-lines generator, empirical
one-sided quotients, and side-effect-free repeated-lag RK4 refinement.  It has
no file-system side effects and never changes a checkpoint model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

try:
    from .latent_integrators import explicit_step, integer_depth, interaction_matrix
except ImportError:  # pragma: no cover - direct script execution on SCNet
    from latent_integrators import explicit_step, integer_depth, interaction_matrix  # type: ignore


RELATIVE_FLOOR = 1e-12
SATURATION_THRESHOLD = 19.999


def _validate_state_batch(states: torch.Tensor, label: str = "states") -> None:
    if states.ndim != 2 or int(states.shape[0]) < 1 or int(states.shape[1]) < 2:
        raise ValueError(
            f"{label} must have nonempty shape (B, N) with N >= 2, "
            f"got {tuple(states.shape)}"
        )
    if not torch.is_floating_point(states):
        raise ValueError(f"{label} must have a floating dtype")


def _encoder_epsilon(model: torch.nn.Module) -> float:
    value = getattr(model, "eps", None)
    if value is None:
        raise TypeError("model must expose the sigmoid/logit encoder epsilon")
    epsilon = float(torch.as_tensor(value).detach().cpu().item())
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("model encoder epsilon must be finite and lie in (0, 0.5)")
    return epsilon


def mesh_l2_norm(values: torch.Tensor, *, length: float) -> torch.Tensor:
    """Return the uniform-periodic-grid L2 norm for every leading sample."""

    _validate_state_batch(values, "values")
    if not math.isfinite(length) or float(length) <= 0:
        raise ValueError("length must be finite and positive")
    dx = float(length) / int(values.shape[-1])
    return torch.sqrt(dx * values.square().sum(dim=1))


def distribution(values: torch.Tensor) -> dict[str, Any]:
    """Summarize one tensor without allowing non-finite values into JSON."""

    flattened = values.detach().reshape(-1).to(torch.float64).cpu()
    finite_mask = torch.isfinite(flattened)
    finite = flattened[finite_mask]
    count = int(flattened.numel())
    if not int(finite.numel()):
        return {
            "count": count,
            "finite_count": 0,
            "nonfinite_count": count,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "min": None,
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.5, 0.9, 0.95], dtype=finite.dtype),
    )
    return {
        "count": count,
        "finite_count": int(finite.numel()),
        "nonfinite_count": count - int(finite.numel()),
        "mean": float(finite.mean().item()),
        "median": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "max": float(finite.max().item()),
        "min": float(finite.min().item()),
    }


def fisher_spectral_generator(
    states: torch.Tensor,
    *,
    length: float,
    diffusivity: float,
    reaction_rate: float,
) -> torch.Tensor:
    """Evaluate the periodic Fourier-collocation Fisher generator."""

    _validate_state_batch(states)
    if not math.isfinite(length) or float(length) <= 0:
        raise ValueError("length must be finite and positive")
    if not math.isfinite(diffusivity) or float(diffusivity) <= 0:
        raise ValueError("diffusivity must be finite and positive")
    if not math.isfinite(reaction_rate) or float(reaction_rate) <= 0:
        raise ValueError("reaction_rate must be finite and positive")
    n_sites = int(states.shape[1])
    modes = torch.arange(
        n_sites // 2 + 1,
        dtype=states.dtype,
        device=states.device,
    )
    wave_numbers = (2.0 * math.pi / float(length)) * modes
    coefficients = torch.fft.rfft(states, dim=-1)
    laplacian = torch.fft.irfft(
        -(wave_numbers.square()) * coefficients,
        n=n_sites,
        dim=-1,
    )
    return float(diffusivity) * laplacian + float(reaction_rate) * states * (
        1.0 - states
    )


def learned_physical_generator(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    conditioning_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``du/dt``, represented state, and latent state for one checkpoint.

    ``represented_state`` is ``decode(encode(states))``.  It makes any encoder
    clamp explicit and keeps the chain-rule vector field aligned with the state
    at which it is evaluated.
    """

    encode = getattr(model, "encode", None)
    decode = getattr(model, "decode", None)
    dynamics = getattr(model, "latent_dynamics", None)
    if not callable(encode) or not callable(decode) or not callable(dynamics):
        raise TypeError("model must expose encode, decode, and latent_dynamics")
    _validate_state_batch(states)
    _encoder_epsilon(model)
    if not math.isfinite(conditioning_time) or float(conditioning_time) <= 0:
        raise ValueError("conditioning_time must be finite and positive")

    latent = encode(states)
    represented = decode(latent)
    interactions = interaction_matrix(model)
    if getattr(model, "latent_dynamics_requires_tau", False):
        latent_rhs = dynamics(latent, interactions, tau=float(conditioning_time))
    else:
        latent_rhs = dynamics(latent, interactions)
    decoder_derivative = represented * (1.0 - represented)
    return decoder_derivative * latent_rhs, represented, latent


def generator_mse_loss(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    conditioning_time: float,
    length: float,
    diffusivity: float,
    reaction_rate: float,
) -> torch.Tensor:
    """Differentiable physical-generator matching loss for Fisher--KPP."""

    learned, represented, _latent = learned_physical_generator(
        model,
        states,
        conditioning_time=conditioning_time,
    )
    reference = fisher_spectral_generator(
        represented,
        length=length,
        diffusivity=diffusivity,
        reaction_rate=reaction_rate,
    )
    return (learned - reference).square().mean()


def _rms(values: torch.Tensor) -> float:
    finite = values.detach().reshape(-1).to(torch.float64).cpu()
    if not bool(torch.isfinite(finite).all().item()):
        raise RuntimeError("RMS input contains a non-finite value")
    return float(torch.sqrt(finite.square().mean()).item())


@torch.no_grad()
def generator_residual_metrics(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    conditioning_time: float,
    length: float,
    diffusivity: float,
    reaction_rate: float,
    batch_size: int,
    relative_floor: float = RELATIVE_FLOOR,
) -> dict[str, Any]:
    """Measure represented-state generator mismatch on one finite collection."""

    _validate_state_batch(states)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(relative_floor) <= 0:
        raise ValueError("relative_floor must be positive")

    residual_norms: list[torch.Tensor] = []
    reference_norms: list[torch.Tensor] = []
    learned_norms: list[torch.Tensor] = []
    relative_norms: list[torch.Tensor] = []
    cosine_values: list[torch.Tensor] = []
    representation_errors: list[torch.Tensor] = []
    latent_abs_values: list[torch.Tensor] = []
    represented_min = math.inf
    represented_max = -math.inf
    raw_min = float(states.min().item())
    raw_max = float(states.max().item())
    raw_bound_violation_count = int(((states < 0.0) | (states > 1.0)).sum().item())

    for start in range(0, int(states.shape[0]), int(batch_size)):
        batch = states[start : start + int(batch_size)]
        learned, represented, latent = learned_physical_generator(
            model,
            batch,
            conditioning_time=float(conditioning_time),
        )
        reference = fisher_spectral_generator(
            represented,
            length=length,
            diffusivity=diffusivity,
            reaction_rate=reaction_rate,
        )
        residual = learned - reference
        residual_norm = mesh_l2_norm(residual, length=length)
        reference_norm = mesh_l2_norm(reference, length=length)
        learned_norm = mesh_l2_norm(learned, length=length)
        inner = (float(length) / int(states.shape[1])) * (learned * reference).sum(
            dim=1
        )
        cosine_denominator = learned_norm * reference_norm
        cosine = torch.where(
            cosine_denominator > float(relative_floor),
            inner / torch.clamp(cosine_denominator, min=float(relative_floor)),
            torch.full_like(inner, float("nan")),
        )
        residual_norms.append(residual_norm.cpu())
        reference_norms.append(reference_norm.cpu())
        learned_norms.append(learned_norm.cpu())
        relative_norms.append(
            (
                residual_norm / torch.clamp(reference_norm, min=float(relative_floor))
            ).cpu()
        )
        cosine_values.append(cosine.cpu())
        representation_errors.append(
            mesh_l2_norm(represented - batch, length=length).cpu()
        )
        latent_abs_values.append(
            latent.abs().reshape(latent.shape[0], -1).max(dim=1).values.cpu()
        )
        represented_min = min(represented_min, float(represented.min().item()))
        represented_max = max(represented_max, float(represented.max().item()))

    residual_tensor = torch.cat(residual_norms)
    reference_tensor = torch.cat(reference_norms)
    learned_tensor = torch.cat(learned_norms)
    relative_tensor = torch.cat(relative_norms)
    cosine_tensor = torch.cat(cosine_values)
    representation_tensor = torch.cat(representation_errors)
    latent_abs_tensor = torch.cat(latent_abs_values)
    nonfinite_events = sum(
        int((~torch.isfinite(tensor)).sum().item())
        for tensor in (
            residual_tensor,
            reference_tensor,
            learned_tensor,
            relative_tensor,
            representation_tensor,
            latent_abs_tensor,
        )
    )
    rms_residual = _rms(residual_tensor)
    rms_reference = _rms(reference_tensor)
    return {
        "sample_count": int(states.shape[0]),
        "residual_l2": distribution(residual_tensor),
        "relative_residual_l2": distribution(relative_tensor),
        "reference_generator_l2": distribution(reference_tensor),
        "learned_generator_l2": distribution(learned_tensor),
        "cosine_alignment": distribution(cosine_tensor),
        "representation_error_l2": distribution(representation_tensor),
        "latent_abs_max": distribution(latent_abs_tensor),
        "raw_state_min": raw_min,
        "raw_state_max": raw_max,
        "raw_bound_violation_count": raw_bound_violation_count,
        "represented_state_min": represented_min,
        "represented_state_max": represented_max,
        "rms_residual_l2": rms_residual,
        "rms_reference_l2": rms_reference,
        "rms_relative_residual": rms_residual
        / max(rms_reference, float(relative_floor)),
        "nonfinite_events": nonfinite_events,
        "saturation_count": int(
            (latent_abs_tensor >= SATURATION_THRESHOLD).sum().item()
        ),
    }


def deterministic_sinusoidal_pair(
    states: torch.Tensor,
    *,
    amplitude: float,
    lower: float = 1e-5,
    upper: float = 1.0 - 1e-5,
) -> torch.Tensor:
    """Create checkpoint-independent local perturbations with deterministic modes."""

    _validate_state_batch(states)
    if not math.isfinite(amplitude) or float(amplitude) <= 0:
        raise ValueError("amplitude must be finite and positive")
    if not (0.0 <= float(lower) < float(upper) <= 1.0):
        raise ValueError("perturbation bounds must satisfy 0 <= lower < upper <= 1")

    batch, n_sites = map(int, states.shape)
    grid = (2.0 * math.pi / n_sites) * torch.arange(
        n_sites,
        device=states.device,
        dtype=states.dtype,
    )
    sample_index = torch.arange(batch, device=states.device)
    modes = (1 + sample_index.remainder(7)).to(states.dtype).reshape(-1, 1)
    phases = (
        2.0 * math.pi * sample_index.remainder(17).to(states.dtype) / 17.0
    ).reshape(-1, 1)
    direction = torch.sin(modes * grid.reshape(1, -1) + phases)
    direction = direction + 0.5 * torch.cos(
        (modes + 1.0) * grid.reshape(1, -1) - phases
    )
    rms = torch.sqrt(direction.square().mean(dim=1, keepdim=True)).clamp_min(1e-12)
    direction = direction / rms
    return torch.clamp(
        states + float(amplitude) * direction,
        min=float(lower),
        max=float(upper),
    )


def frozen_pair(states: torch.Tensor, family: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one of the preregistered deterministic OSL pair families."""

    if states.ndim != 2 or int(states.shape[0]) < 2:
        raise ValueError("pair diagnostics require at least two rank-2 states")
    normalized = str(family).strip().lower()
    if normalized == "cyclic_cross_sample":
        return states, torch.roll(states, shifts=-1, dims=0)
    amplitudes = {
        "sinusoidal_0.001": 0.001,
        "sinusoidal_0.01": 0.01,
    }
    if normalized not in amplitudes:
        raise ValueError(f"unsupported pair family: {family!r}")
    return states, deterministic_sinusoidal_pair(
        states, amplitude=amplitudes[normalized]
    )


def one_sided_quotient(
    first_rhs: torch.Tensor,
    second_rhs: torch.Tensor,
    first_state: torch.Tensor,
    second_state: torch.Tensor,
    *,
    length: float,
    separation_floor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empirical one-sided quotients and pair distances."""

    if not (
        first_rhs.shape == second_rhs.shape == first_state.shape == second_state.shape
    ):
        raise ValueError("states and vector fields must have identical shapes")
    _validate_state_batch(first_state, "first_state")
    if not math.isfinite(length) or float(length) <= 0:
        raise ValueError("length must be finite and positive")
    if separation_floor is None:
        finfo = torch.finfo(first_state.dtype)
        separation_floor = max(
            float(finfo.tiny),
            float(finfo.eps) ** 2 * max(float(length), 1.0),
        )
    if not math.isfinite(separation_floor) or float(separation_floor) <= 0:
        raise ValueError("separation_floor must be finite and positive")
    delta = first_state - second_state
    delta_rhs = first_rhs - second_rhs
    dx = float(length) / int(delta.shape[-1])
    denominator = dx * delta.square().sum(dim=1)
    if bool((denominator <= float(separation_floor)).any().item()):
        raise ValueError("one-sided quotient pair has zero numerical separation")
    numerator = dx * (delta_rhs * delta).sum(dim=1)
    return numerator / denominator, torch.sqrt(denominator)


@torch.no_grad()
def stability_pair_metrics(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    pair_family: str,
    conditioning_time: float,
    length: float,
    diffusivity: float,
    reaction_rate: float,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate reference and learned OSL quotients on exactly the same pairs."""

    _validate_state_batch(states)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    first, second = frozen_pair(states, pair_family)
    reference_values: list[torch.Tensor] = []
    learned_values: list[torch.Tensor] = []
    pair_distances: list[torch.Tensor] = []
    for start in range(0, int(states.shape[0]), int(batch_size)):
        first_batch = first[start : start + int(batch_size)]
        second_batch = second[start : start + int(batch_size)]

        first_learned, first_represented, _first_latent = learned_physical_generator(
            model,
            first_batch,
            conditioning_time=conditioning_time,
        )
        second_learned, second_represented, _second_latent = learned_physical_generator(
            model,
            second_batch,
            conditioning_time=conditioning_time,
        )
        first_reference_state = first_represented.to(torch.float64)
        second_reference_state = second_represented.to(torch.float64)
        first_reference = fisher_spectral_generator(
            first_reference_state,
            length=length,
            diffusivity=diffusivity,
            reaction_rate=reaction_rate,
        )
        second_reference = fisher_spectral_generator(
            second_reference_state,
            length=length,
            diffusivity=diffusivity,
            reaction_rate=reaction_rate,
        )
        reference_quotient, distance = one_sided_quotient(
            first_reference,
            second_reference,
            first_reference_state,
            second_reference_state,
            length=length,
        )

        learned_quotient, _learned_distance = one_sided_quotient(
            first_learned,
            second_learned,
            first_represented,
            second_represented,
            length=length,
        )
        reference_values.append(reference_quotient.cpu())
        learned_values.append(learned_quotient.cpu())
        pair_distances.append(distance.cpu())

    reference_tensor = torch.cat(reference_values)
    learned_tensor = torch.cat(learned_values)
    distance_tensor = torch.cat(pair_distances)
    return {
        "sample_count": int(states.shape[0]),
        "pair_family": pair_family,
        "reference_quotient": distribution(reference_tensor),
        "learned_quotient": distribution(learned_tensor),
        "pair_distance_l2": distribution(distance_tensor),
        "nonfinite_events": int((~torch.isfinite(reference_tensor)).sum().item())
        + int((~torch.isfinite(learned_tensor)).sum().item())
        + int((~torch.isfinite(distance_tensor)).sum().item()),
    }


def _latent_rhs(
    model: torch.nn.Module,
    interactions: torch.Tensor,
    conditioning_time: float,
):
    dynamics = getattr(model, "latent_dynamics", None)
    if not callable(dynamics):
        raise TypeError("model must expose latent_dynamics")
    if getattr(model, "latent_dynamics_requires_tau", False):

        def rhs(value: torch.Tensor) -> torch.Tensor:
            return dynamics(value, interactions, tau=float(conditioning_time))

    else:

        def rhs(value: torch.Tensor) -> torch.Tensor:
            return dynamics(value, interactions)

    return rhs


@torch.no_grad()
def repeated_lag_snapshots(
    model: torch.nn.Module,
    initial_states: torch.Tensor,
    *,
    lag: float,
    horizons: Sequence[float],
    substeps_per_call: int,
) -> tuple[dict[float, torch.Tensor], dict[str, Any]]:
    """Advance repeated lag calls and capture selected physical endpoints."""

    if int(substeps_per_call) <= 0:
        raise ValueError("substeps_per_call must be positive")
    _validate_state_batch(initial_states, "initial_states")
    if not math.isfinite(lag) or float(lag) <= 0:
        raise ValueError("lag must be finite and positive")
    ordered_horizons = tuple(sorted(set(float(value) for value in horizons)))
    if not ordered_horizons:
        raise ValueError("at least one horizon is required")
    depth_to_horizon = {
        integer_depth(horizon, float(lag)): horizon for horizon in ordered_horizons
    }
    if len(depth_to_horizon) != len(ordered_horizons):
        raise ValueError("horizons must map to distinct repeated-lag depths")

    encode = getattr(model, "encode", None)
    decode = getattr(model, "decode", None)
    if not callable(encode) or not callable(decode):
        raise TypeError("model must expose encode and decode")
    interactions = interaction_matrix(model)
    state = initial_states
    snapshots: dict[float, torch.Tensor] = {}
    saturation_count = 0
    observed_latent_values = 0
    max_abs_latent = 0.0
    encoder_epsilon = _encoder_epsilon(model)
    encoder_clamp_activation_count = 0
    observed_decoded_values = 0
    reencoding_abs_max = 0.0
    reencoding_squared_sum = 0.0
    reencoding_value_count = 0
    max_depth = max(depth_to_horizon)

    for depth in range(1, max_depth + 1):
        latent = encode(state)
        rhs = _latent_rhs(model, interactions, float(lag))
        dt = torch.as_tensor(
            float(lag) / int(substeps_per_call),
            dtype=latent.dtype,
            device=latent.device,
        )
        initial_abs = latent.abs()
        saturation_count += int((initial_abs >= SATURATION_THRESHOLD).sum().item())
        observed_latent_values += int(initial_abs.numel())
        max_abs_latent = max(max_abs_latent, float(initial_abs.max().item()))
        for _ in range(int(substeps_per_call)):
            latent = explicit_step(rhs, latent, dt, "rk4")
            latent = torch.clamp(latent, -20.0, 20.0)
            absolute = latent.abs()
            saturation_count += int((absolute >= SATURATION_THRESHOLD).sum().item())
            observed_latent_values += int(absolute.numel())
            max_abs_latent = max(max_abs_latent, float(absolute.max().item()))
        state = decode(latent)
        encoder_clamp_activation_count += int(
            ((state <= encoder_epsilon) | (state >= 1.0 - encoder_epsilon)).sum().item()
        )
        observed_decoded_values += int(state.numel())
        reencoded = encode(state)
        reencoding_difference = reencoded - latent
        reencoding_abs_max = max(
            reencoding_abs_max,
            float(reencoding_difference.abs().max().item()),
        )
        reencoding_squared_sum += float(reencoding_difference.square().sum().item())
        reencoding_value_count += int(reencoding_difference.numel())
        if depth in depth_to_horizon:
            snapshots[depth_to_horizon[depth]] = state.detach().clone()

    total_substeps = max_depth * int(substeps_per_call)
    return snapshots, {
        "lag": float(lag),
        "max_horizon": max(ordered_horizons),
        "max_depth": max_depth,
        "substeps_per_call": int(substeps_per_call),
        "total_substeps_to_max_horizon": total_substeps,
        "total_rhs_evaluations_to_max_horizon": 4 * total_substeps,
        "observed_latent_values": observed_latent_values,
        "saturation_count": saturation_count,
        "saturation_fraction": (
            saturation_count / observed_latent_values if observed_latent_values else 0.0
        ),
        "max_abs_latent": max_abs_latent,
        "encoder_epsilon": encoder_epsilon,
        "observed_decoded_values": observed_decoded_values,
        "encoder_clamp_activation_count": encoder_clamp_activation_count,
        "encoder_clamp_activation_fraction": (
            encoder_clamp_activation_count / observed_decoded_values
            if observed_decoded_values
            else 0.0
        ),
        "reencoding_abs_max": reencoding_abs_max,
        "reencoding_rms": math.sqrt(reencoding_squared_sum / reencoding_value_count)
        if reencoding_value_count
        else 0.0,
    }


@torch.no_grad()
def autonomy_sensitivity_metrics(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    conditioning_times: Sequence[float],
    length: float,
) -> dict[str, Any]:
    """Measure physical-generator dependence on nominal query-time labels."""

    normalized = tuple(float(value) for value in conditioning_times)
    if len(normalized) != 2:
        raise ValueError("autonomy sensitivity requires exactly two query times")
    first, _represented_first, _latent_first = learned_physical_generator(
        model,
        states,
        conditioning_time=normalized[0],
    )
    second, _represented_second, _latent_second = learned_physical_generator(
        model,
        states,
        conditioning_time=normalized[1],
    )
    differences = mesh_l2_norm(first - second, length=length)
    return {
        "conditioning_times": list(normalized),
        "difference_l2": distribution(differences),
        "max_difference_l2": float(differences.max().item()),
    }


@torch.no_grad()
def refinement_metrics(
    model: torch.nn.Module,
    initial_states: torch.Tensor,
    cache_reference: torch.Tensor,
    *,
    lag: float,
    horizon: float,
    substeps: Sequence[int],
    finest_substeps: int,
    production_substeps: int,
    length: float,
    relative_floor: float = RELATIVE_FLOOR,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare repeated-lag RK4 levels with one frozen finest surrogate."""

    _validate_state_batch(initial_states, "initial_states")
    _validate_state_batch(cache_reference, "cache_reference")
    if initial_states.shape != cache_reference.shape:
        raise ValueError(
            "initial_states and cache_reference must have identical shapes"
        )
    counts = tuple(sorted(set(int(value) for value in substeps)))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("refinement substeps must be positive")
    if int(finest_substeps) <= counts[-1]:
        raise ValueError("finest_substeps must exceed every reported refinement level")
    if int(production_substeps) not in counts:
        raise ValueError("production_substeps must be one reported refinement level")
    depth = integer_depth(float(horizon), float(lag))

    outputs: dict[int, torch.Tensor] = {}
    work: dict[int, dict[str, Any]] = {}
    for count in (*counts, int(finest_substeps)):
        snapshots, execution = repeated_lag_snapshots(
            model,
            initial_states,
            lag=float(lag),
            horizons=(float(horizon),),
            substeps_per_call=int(count),
        )
        outputs[int(count)] = snapshots[float(horizon)]
        work[int(count)] = execution

    finest = outputs[int(finest_substeps)]
    finest_norm = mesh_l2_norm(finest, length=length)
    rows: list[dict[str, Any]] = []
    errors: dict[int, float] = {}
    for count in counts:
        output = outputs[count]
        difference = mesh_l2_norm(output - finest, length=length)
        relative = difference / torch.clamp(
            finest_norm,
            min=float(relative_floor),
        )
        cache_difference = mesh_l2_norm(output - cache_reference, length=length)
        errors[count] = _rms(difference)
        rows.append(
            {
                "lag": float(lag),
                "horizon": float(horizon),
                "composition_depth": depth,
                "substeps_per_call": count,
                "finest_substeps_per_call": int(finest_substeps),
                "total_substeps": depth * count,
                "total_rhs_evaluations": 4 * depth * count,
                "learned_flow_difference_vs_finest_l2": distribution(difference),
                "relative_learned_flow_difference_vs_finest_l2": distribution(relative),
                "rms_learned_flow_difference_vs_finest_l2": errors[count],
                "learned_prediction_discrepancy_vs_cache_reference_l2": distribution(
                    cache_difference
                ),
                "rms_learned_prediction_discrepancy_vs_cache_reference_l2": _rms(
                    cache_difference
                ),
                "saturation_count": work[count]["saturation_count"],
                "saturation_fraction": work[count]["saturation_fraction"],
                "max_abs_latent": work[count]["max_abs_latent"],
                "encoder_clamp_activation_count": work[count][
                    "encoder_clamp_activation_count"
                ],
                "encoder_clamp_activation_fraction": work[count][
                    "encoder_clamp_activation_fraction"
                ],
                "reencoding_abs_max": work[count]["reencoding_abs_max"],
                "reencoding_rms": work[count]["reencoding_rms"],
                "observed_order_to_next": None,
                "next_substeps_per_call": None,
            }
        )

    for index in range(len(rows) - 1):
        coarse = counts[index]
        fine = counts[index + 1]
        coarse_error = errors[coarse]
        fine_error = errors[fine]
        if coarse_error > float(relative_floor) and fine_error > float(relative_floor):
            rows[index]["observed_order_to_next"] = math.log(
                coarse_error / fine_error
            ) / math.log(fine / coarse)
            rows[index]["next_substeps_per_call"] = fine

    production = outputs[int(production_substeps)]
    learned_flow_difference = mesh_l2_norm(production - finest, length=length)
    cache_discrepancy = mesh_l2_norm(production - cache_reference, length=length)
    rms_learned_flow_difference = _rms(learned_flow_difference)
    rms_cache_discrepancy = _rms(cache_discrepancy)
    numerical_to_cache_ratio = rms_learned_flow_difference / max(
        rms_cache_discrepancy, float(relative_floor)
    )
    summary = {
        "lag": float(lag),
        "horizon": float(horizon),
        "composition_depth": depth,
        "sample_count": int(initial_states.shape[0]),
        "production_substeps_per_call": int(production_substeps),
        "finest_substeps_per_call": int(finest_substeps),
        "production_rms_learned_flow_difference_vs_finest_l2": (
            rms_learned_flow_difference
        ),
        "production_rms_discrepancy_vs_cache_reference_l2": rms_cache_discrepancy,
        "learned_flow_numerical_to_cache_discrepancy_ratio": numerical_to_cache_ratio,
        "learned_flow_numerical_confounding_threshold": 0.10,
        "learned_flow_numerical_confounding": numerical_to_cache_ratio > 0.10,
        "finest_saturation_count": work[int(finest_substeps)]["saturation_count"],
        "finest_saturation_fraction": work[int(finest_substeps)]["saturation_fraction"],
        "finest_max_abs_latent": work[int(finest_substeps)]["max_abs_latent"],
        "finest_encoder_clamp_activation_count": work[int(finest_substeps)][
            "encoder_clamp_activation_count"
        ],
        "finest_encoder_clamp_activation_fraction": work[int(finest_substeps)][
            "encoder_clamp_activation_fraction"
        ],
        "finest_reencoding_abs_max": work[int(finest_substeps)]["reencoding_abs_max"],
        "finest_reencoding_rms": work[int(finest_substeps)]["reencoding_rms"],
    }
    return summary, rows


def snapshot_states(
    cache: Mapping[str, Any],
    *,
    time_value: float,
    reference_dt: float,
    sample_count: int,
) -> torch.Tensor:
    """Stack one exact stored reference-cache snapshot, including time zero."""

    ratio = float(time_value) / float(reference_dt)
    index = int(round(ratio))
    if not math.isclose(ratio, index, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("snapshot time must align with the reference cache grid")
    trajectories = cache["test_trajs"]
    return torch.stack(
        [trajectories[item][1][index] for item in range(int(sample_count))]
    )


__all__ = [
    "RELATIVE_FLOOR",
    "SATURATION_THRESHOLD",
    "autonomy_sensitivity_metrics",
    "deterministic_sinusoidal_pair",
    "distribution",
    "fisher_spectral_generator",
    "frozen_pair",
    "generator_residual_metrics",
    "learned_physical_generator",
    "mesh_l2_norm",
    "one_sided_quotient",
    "refinement_metrics",
    "repeated_lag_snapshots",
    "snapshot_states",
    "stability_pair_metrics",
]
