from __future__ import annotations

import argparse
from pathlib import Path

from atoms.config import DEFAULT_SIMULATION, SPECIES_BY_KEY
from atoms.physics import run_simulation
from atoms.plotting import plot_phase_space, plot_survival, plot_temperature, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate non-adiabatic atom shuttling in optical tweezers.")
    parser.add_argument(
        "--species",
        nargs="+",
        choices=sorted(SPECIES_BY_KEY.keys()),
        default=sorted(SPECIES_BY_KEY.keys()),
        help="Species keys to simulate.",
    )
    parser.add_argument("--trajectories", type=int, default=DEFAULT_SIMULATION.trajectories)
    parser.add_argument("--time-step-s", type=float, default=DEFAULT_SIMULATION.time_step_s)
    parser.add_argument("--initial-temperature-k", type=float, default=DEFAULT_SIMULATION.initial_temperature_k)
    parser.add_argument("--distance-m", type=float, default=DEFAULT_SIMULATION.trajectory.distance_m)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_SIMULATION.trajectory.duration_s)
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
    parser.add_argument(
        "--phase-space-samples",
        type=int,
        default=DEFAULT_SIMULATION.phase_space_samples,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION.random_seed)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "default_run")
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
        phase_space_samples=args.phase_space_samples,
        random_seed=args.seed,
        distance_m=args.distance_m,
        duration_s=args.duration_s,
        saturation_parameter=args.saturation_parameter,
        detuning_in_gamma=args.detuning_in_gamma,
        beam_waist_m=args.beam_waist_m,
        peak_intensity_w_m2=args.peak_intensity_w_m2,
    )

    results = []
    for index, key in enumerate(args.species):
        species = SPECIES_BY_KEY[key]
        result = run_simulation(species, config.with_overrides(random_seed=args.seed + index))
        results.append(result)

    phase_space_path = plot_phase_space(results, output_dir)
    temperature_path = plot_temperature(results, output_dir)
    survival_path = plot_survival(results, output_dir)
    summary_path = write_summary(results, config, output_dir)

    print(f"Wrote {phase_space_path}")
    print(f"Wrote {temperature_path}")
    print(f"Wrote {survival_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
