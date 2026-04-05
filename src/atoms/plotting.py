from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from atoms.config import SimulationConfig
from atoms.physics import SimulationResult, diffusion_coefficient

if TYPE_CHECKING:
    from atoms.speed_scan import SurvivalSweepResult


def _time_microseconds(values_s: np.ndarray) -> np.ndarray:
    return values_s * 1.0e6


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if np.isfinite(value):
        return float(value)
    return None


def write_summary(results: list[SimulationResult], config: SimulationConfig, output_dir: Path) -> Path:
    summary = {
        "simulation": {
            "trajectories": config.trajectories,
            "time_step_s": config.time_step_s,
            "initial_temperature_k": config.initial_temperature_k,
            "distance_m": config.trajectory.distance_m,
            "duration_s": config.trajectory.duration_s,
            "beam_waist_m": config.tweezer.beam_waist_m,
            "peak_intensity_w_m2": config.tweezer.peak_intensity_w_m2,
            "saturation_parameter": config.cooling.saturation_parameter,
            "detuning_in_gamma": config.cooling.detuning_in_gamma,
            "random_seed": config.random_seed,
        },
        "species": [],
    }
    for result in results:
        detuning = config.cooling.detuning_rad_s(result.species)
        summary["species"].append(
            {
                "key": result.species.key,
                "label": result.species.label,
                "final_survival_probability": float(result.survival_probability[-1]),
                "max_kinetic_temperature_k": _finite_or_none(np.nanmax(result.kinetic_temperature_k)),
                "final_kinetic_temperature_k": _finite_or_none(result.kinetic_temperature_k[-1]),
                "final_trapped_kinetic_temperature_k": _finite_or_none(result.trapped_kinetic_temperature_k[-1]),
                "trap_depth_k": result.species.trap_depth_k,
                "gamma_rad_s": result.species.gamma_rad_s,
                "effective_polarizability_si": config.tweezer.effective_polarizability_si(result.species),
                "diffusion_coefficient_kg2_m2_s3": diffusion_coefficient(
                    result.species,
                    config.cooling.saturation_parameter,
                    detuning,
                ),
                "recoil_energy_k": constants.hbar**2
                * result.species.wave_number_m**2
                / (2.0 * result.species.mass_kg * constants.k),
            }
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    return summary_path


def plot_phase_space(results: list[SimulationResult], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 4.5), sharex=False, sharey=False)
    axes = np.atleast_1d(axes)
    for axis, result in zip(axes, results):
        sample_count = min(result.config.phase_space_samples, result.relative_position_m.shape[0])
        for index in range(sample_count):
            axis.plot(
                result.relative_position_m[index] * 1.0e6,
                result.momentum_kg_m_s[index] / (constants.hbar * result.species.wave_number_m),
                alpha=0.35,
                linewidth=0.9,
            )
        axis.set_title(result.species.label)
        axis.set_xlabel("Relative Position (um)")
        axis.set_ylabel("Momentum (hbar k)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "phase_space.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_temperature(results: list[SimulationResult], output_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    for result in results:
        axis.plot(
            _time_microseconds(result.time_s),
            result.kinetic_temperature_k * 1.0e6,
            label=result.species.label,
            linewidth=2.0,
        )
    axis.set_xlabel("Time (us)")
    axis.set_ylabel("Kinetic Temperature (uK)")
    axis.set_title("Trap-Frame Kinetic Temperature")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "kinetic_temperature.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_survival(results: list[SimulationResult], output_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    for result in results:
        axis.plot(
            _time_microseconds(result.time_s),
            result.survival_probability,
            label=result.species.label,
            linewidth=2.0,
        )
    axis.set_xlabel("Time (us)")
    axis.set_ylabel("Survival Probability")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Atom Survival During Shuttling")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "survival_probability.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_speed_scan_summary(results: list["SurvivalSweepResult"], output_dir: Path) -> Path:
    summary = {"species": []}
    for result in results:
        threshold = result.threshold
        summary["species"].append(
            {
                "key": result.species.key,
                "label": result.species.label,
                "target_survival_probability": result.target_survival_probability,
                "threshold_status": threshold.status,
                "speed_limit_m_s": _finite_or_none(threshold.speed_limit_m_s),
                "bracket_lower_speed_m_s": _finite_or_none(threshold.lower_speed_m_s),
                "bracket_upper_speed_m_s": _finite_or_none(threshold.upper_speed_m_s),
                "bracket_lower_survival_probability": _finite_or_none(threshold.lower_survival_probability),
                "bracket_upper_survival_probability": _finite_or_none(threshold.upper_survival_probability),
                "sweep_points": [
                    {
                        "duration_s": point.duration_s,
                        "average_speed_m_s": point.average_speed_m_s,
                        "final_survival_probability": point.final_survival_probability,
                        "survival_standard_error": point.survival_standard_error,
                        "peak_kinetic_temperature_k": point.peak_kinetic_temperature_k,
                    }
                    for point in result.points
                ],
            }
        )
    path = output_dir / "speed_scan_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    return path


def plot_survival_vs_speed(results: list["SurvivalSweepResult"], output_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, result in enumerate(results):
        speeds = result.average_speeds_m_s
        survivals = result.final_survival_probabilities
        errors = result.survival_standard_errors
        color = colors[index % len(colors)] if colors else None
        axis.plot(
            speeds,
            survivals,
            marker="o",
            linewidth=2.0,
            label=result.species.label,
            color=color,
        )
        axis.fill_between(
            speeds,
            np.clip(survivals - errors, 0.0, 1.0),
            np.clip(survivals + errors, 0.0, 1.0),
            alpha=0.18,
            color=color,
        )
        if result.threshold.speed_limit_m_s is not None:
            axis.axvline(result.threshold.speed_limit_m_s, linestyle="--", linewidth=1.2, alpha=0.7, color=color)
    if results:
        axis.axhline(results[0].target_survival_probability, linestyle=":", linewidth=1.4, color="black", alpha=0.7)
    axis.set_xlabel("Average Shuttle Speed (m/s)")
    axis.set_ylabel("Final Survival Probability")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Survival Versus Shuttle Speed")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "survival_vs_speed.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
