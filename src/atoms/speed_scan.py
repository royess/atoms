from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path

import numpy as np

from atoms.config import DEFAULT_SIMULATION, AtomicSpecies, SPECIES_BY_KEY, SimulationConfig
from atoms.physics import run_simulation
from atoms.plotting import plot_survival_vs_speed, write_speed_scan_summary


@dataclass(frozen=True)
class SurvivalSweepPoint:
    duration_s: float
    average_speed_m_s: float
    final_survival_probability: float
    survival_standard_error: float
    peak_kinetic_temperature_k: float


@dataclass(frozen=True)
class SpeedThresholdEstimate:
    target_survival_probability: float
    status: str
    speed_limit_m_s: float | None
    lower_speed_m_s: float | None
    upper_speed_m_s: float | None
    lower_survival_probability: float | None
    upper_survival_probability: float | None


@dataclass(frozen=True)
class SurvivalSweepResult:
    species: AtomicSpecies
    base_config: SimulationConfig
    target_survival_probability: float
    points: tuple[SurvivalSweepPoint, ...]
    threshold: SpeedThresholdEstimate

    @property
    def durations_s(self) -> np.ndarray:
        return np.array([point.duration_s for point in self.points], dtype=float)

    @property
    def average_speeds_m_s(self) -> np.ndarray:
        return np.array([point.average_speed_m_s for point in self.points], dtype=float)

    @property
    def final_survival_probabilities(self) -> np.ndarray:
        return np.array([point.final_survival_probability for point in self.points], dtype=float)

    @property
    def survival_standard_errors(self) -> np.ndarray:
        return np.array([point.survival_standard_error for point in self.points], dtype=float)


def average_shuttle_speed(distance_m: float, duration_s: float) -> float:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    return distance_m / duration_s


def survival_standard_error(final_survival_probability: float, trajectories: int) -> float:
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    probability = np.clip(final_survival_probability, 0.0, 1.0)
    return float(np.sqrt(probability * (1.0 - probability) / trajectories))


def build_duration_grid(
    duration_min_s: float,
    duration_max_s: float,
    count: int,
    spacing: str = "log",
) -> np.ndarray:
    if duration_min_s <= 0.0 or duration_max_s <= 0.0:
        raise ValueError("durations must be positive")
    if duration_max_s < duration_min_s:
        raise ValueError("duration_max_s must be at least duration_min_s")
    if count < 2:
        raise ValueError("count must be at least 2")
    if spacing == "log":
        return np.geomspace(duration_min_s, duration_max_s, count)
    if spacing == "linear":
        return np.linspace(duration_min_s, duration_max_s, count)
    raise ValueError(f"unsupported spacing: {spacing}")


def monotone_survival_curve(final_survival_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(final_survival_probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("final_survival_probabilities must be one-dimensional")
    return np.minimum.accumulate(probabilities)


def estimate_speed_threshold(
    average_speeds_m_s: np.ndarray,
    final_survival_probabilities: np.ndarray,
    target_survival_probability: float,
) -> SpeedThresholdEstimate:
    speeds = np.asarray(average_speeds_m_s, dtype=float)
    survivals = np.asarray(final_survival_probabilities, dtype=float)
    if speeds.ndim != 1 or survivals.ndim != 1 or speeds.shape != survivals.shape:
        raise ValueError("average_speeds_m_s and final_survival_probabilities must be one-dimensional with equal length")
    if speeds.size < 2:
        raise ValueError("at least two sweep points are required")
    if not 0.0 <= target_survival_probability <= 1.0:
        raise ValueError("target_survival_probability must lie in [0, 1]")
    order = np.argsort(speeds)
    speeds = speeds[order]
    survivals = survivals[order]
    monotone_survivals = monotone_survival_curve(survivals)

    if monotone_survivals[0] < target_survival_probability:
        return SpeedThresholdEstimate(
            target_survival_probability=target_survival_probability,
            status="below_sampled_range",
            speed_limit_m_s=None,
            lower_speed_m_s=None,
            upper_speed_m_s=speeds[0],
            lower_survival_probability=None,
            upper_survival_probability=survivals[0],
        )
    if monotone_survivals[-1] >= target_survival_probability:
        return SpeedThresholdEstimate(
            target_survival_probability=target_survival_probability,
            status="above_sampled_range",
            speed_limit_m_s=None,
            lower_speed_m_s=speeds[-1],
            upper_speed_m_s=None,
            lower_survival_probability=survivals[-1],
            upper_survival_probability=None,
        )

    crossing_index = int(np.argmax(monotone_survivals < target_survival_probability))
    lower_index = crossing_index - 1
    upper_index = crossing_index
    speed_low = speeds[lower_index]
    speed_high = speeds[upper_index]
    survival_low = monotone_survivals[lower_index]
    survival_high = monotone_survivals[upper_index]

    if np.isclose(survival_low, target_survival_probability):
        speed_limit_m_s = float(speed_low)
    elif np.isclose(survival_high, target_survival_probability):
        speed_limit_m_s = float(speed_high)
    elif np.isclose(survival_low, survival_high):
        speed_limit_m_s = float(0.5 * (speed_low + speed_high))
    else:
        speed_limit_m_s = float(
            speed_low
            + (target_survival_probability - survival_low) * (speed_high - speed_low) / (survival_high - survival_low)
        )

    return SpeedThresholdEstimate(
        target_survival_probability=target_survival_probability,
        status="bracketed",
        speed_limit_m_s=speed_limit_m_s,
        lower_speed_m_s=float(speed_low),
        upper_speed_m_s=float(speed_high),
        lower_survival_probability=float(survivals[lower_index]),
        upper_survival_probability=float(survivals[upper_index]),
    )


def run_survival_speed_sweep(
    species: AtomicSpecies,
    base_config: SimulationConfig,
    durations_s: np.ndarray | list[float],
    *,
    target_survival_probability: float = 0.5,
) -> SurvivalSweepResult:
    durations = np.asarray(durations_s, dtype=float)
    if durations.ndim != 1 or durations.size < 2:
        raise ValueError("durations_s must be a one-dimensional array with at least two durations")
    if np.any(durations <= 0.0):
        raise ValueError("durations must be positive")
    if not 0.0 <= target_survival_probability <= 1.0:
        raise ValueError("target_survival_probability must lie in [0, 1]")

    points: list[SurvivalSweepPoint] = []
    for index, duration_s in enumerate(durations):
        config = base_config.with_overrides(duration_s=float(duration_s), random_seed=base_config.random_seed + index)
        result = run_simulation(species, config)
        final_survival = float(result.survival_probability[-1])
        points.append(
            SurvivalSweepPoint(
                duration_s=float(duration_s),
                average_speed_m_s=average_shuttle_speed(config.trajectory.distance_m, float(duration_s)),
                final_survival_probability=final_survival,
                survival_standard_error=survival_standard_error(final_survival, config.trajectories),
                peak_kinetic_temperature_k=float(np.nanmax(result.kinetic_temperature_k)),
            )
        )

    points = sorted(points, key=lambda point: point.average_speed_m_s)
    threshold = estimate_speed_threshold(
        np.array([point.average_speed_m_s for point in points], dtype=float),
        np.array([point.final_survival_probability for point in points], dtype=float),
        target_survival_probability,
    )
    return SurvivalSweepResult(
        species=species,
        base_config=base_config,
        target_survival_probability=target_survival_probability,
        points=tuple(points),
        threshold=threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep shuttle duration and estimate a survival-based speed limit.")
    parser.add_argument(
        "--species",
        nargs="+",
        choices=sorted(SPECIES_BY_KEY.keys()),
        default=sorted(SPECIES_BY_KEY.keys()),
        help="Species keys to scan.",
    )
    parser.add_argument("--trajectories", type=int, default=DEFAULT_SIMULATION.trajectories)
    parser.add_argument("--time-step-s", type=float, default=DEFAULT_SIMULATION.time_step_s)
    parser.add_argument("--initial-temperature-k", type=float, default=DEFAULT_SIMULATION.initial_temperature_k)
    parser.add_argument("--distance-m", type=float, default=DEFAULT_SIMULATION.trajectory.distance_m)
    parser.add_argument(
        "--durations-s",
        nargs="*",
        type=float,
        default=None,
        help="Explicit duration values to scan. If omitted, a duration grid is generated from the range flags.",
    )
    parser.add_argument("--duration-min-s", type=float, default=5.0e-6)
    parser.add_argument("--duration-max-s", type=float, default=1.0e-3)
    parser.add_argument("--num-durations", type=int, default=16)
    parser.add_argument("--duration-spacing", choices=("log", "linear"), default="log")
    parser.add_argument("--target-survival", type=float, default=0.5)
    parser.add_argument(
        "--saturation-parameter",
        type=float,
        default=DEFAULT_SIMULATION.cooling.saturation_parameter,
    )
    parser.add_argument(
        "--detuning-in-gamma",
        type=float,
        default=DEFAULT_SIMULATION.cooling.detuning_in_gamma,
    )
    parser.add_argument("--beam-waist-m", type=float, default=DEFAULT_SIMULATION.tweezer.beam_waist_m)
    parser.add_argument(
        "--peak-intensity-w-m2",
        type=float,
        default=DEFAULT_SIMULATION.tweezer.peak_intensity_w_m2,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION.random_seed)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "speed_scan")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_SIMULATION.with_overrides(
        trajectories=args.trajectories,
        time_step_s=args.time_step_s,
        initial_temperature_k=args.initial_temperature_k,
        random_seed=args.seed,
        distance_m=args.distance_m,
        saturation_parameter=args.saturation_parameter,
        detuning_in_gamma=args.detuning_in_gamma,
        beam_waist_m=args.beam_waist_m,
        peak_intensity_w_m2=args.peak_intensity_w_m2,
    )
    durations = (
        np.array(args.durations_s, dtype=float)
        if args.durations_s
        else build_duration_grid(args.duration_min_s, args.duration_max_s, args.num_durations, args.duration_spacing)
    )

    results = []
    for index, key in enumerate(args.species):
        species = SPECIES_BY_KEY[key]
        result = run_survival_speed_sweep(
            species,
            config.with_overrides(random_seed=args.seed + index * 1000),
            durations,
            target_survival_probability=args.target_survival,
        )
        results.append(result)

    plot_path = plot_survival_vs_speed(results, output_dir)
    summary_path = write_speed_scan_summary(results, output_dir)

    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    for result in results:
        threshold = result.threshold
        if threshold.speed_limit_m_s is None:
            print(f"{result.species.label}: no threshold crossing in sampled range ({threshold.status})")
        else:
            print(
                f"{result.species.label}: {result.target_survival_probability:.2f} survival speed limit ~ "
                f"{threshold.speed_limit_m_s:.4f} m/s"
            )


if __name__ == "__main__":
    main()
