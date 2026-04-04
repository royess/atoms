from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from atoms.config import SimulationConfig
from atoms.physics import SimulationResult, diffusion_coefficient


def _time_microseconds(values_s: np.ndarray) -> np.ndarray:
    return values_s * 1.0e6


def _finite_or_none(value: float) -> float | None:
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
