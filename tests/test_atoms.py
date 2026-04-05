from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import constants

from atoms.config import DEFAULT_SIMULATION, RB87, YB171
from atoms.physics import cooling_force, diffusion_coefficient, dipole_force, run_simulation
from atoms.speed_scan import estimate_speed_threshold, run_survival_speed_sweep
from atoms.trajectory import build_time_grid, minimum_jerk_trajectory


class ConfigTests(unittest.TestCase):
    def test_species_constants_are_converted_to_si(self) -> None:
        self.assertAlmostEqual(RB87.mass_kg, 86.909 * constants.atomic_mass)
        self.assertAlmostEqual(YB171.mass_kg, 170.936 * constants.atomic_mass)
        self.assertAlmostEqual(RB87.trap_depth_j, constants.k * 1.0e-3)
        self.assertAlmostEqual(YB171.trap_depth_j, constants.k * 0.1e-3)
        self.assertAlmostEqual(RB87.saturation_intensity_w_m2, 16.7)
        self.assertAlmostEqual(YB171.saturation_intensity_w_m2, 1.4)


class TrajectoryTests(unittest.TestCase):
    def test_minimum_jerk_boundary_conditions(self) -> None:
        config = DEFAULT_SIMULATION.trajectory
        time_s = build_time_grid(config.duration_s, 1.0e-6)
        position_m, velocity_m_s, acceleration_m_s2 = minimum_jerk_trajectory(config, time_s)

        self.assertAlmostEqual(position_m[0], 0.0)
        self.assertAlmostEqual(position_m[-1], config.distance_m)
        self.assertAlmostEqual(velocity_m_s[0], 0.0)
        self.assertAlmostEqual(velocity_m_s[-1], 0.0)
        self.assertAlmostEqual(acceleration_m_s2[0], 0.0)
        self.assertAlmostEqual(acceleration_m_s2[-1], 0.0)


class PhysicsHelperTests(unittest.TestCase):
    def test_cooling_force_is_odd_in_velocity(self) -> None:
        detuning = DEFAULT_SIMULATION.cooling.detuning_rad_s(RB87)
        velocities = np.linspace(-0.2, 0.2, 9)
        forces = cooling_force(RB87, DEFAULT_SIMULATION.cooling.saturation_parameter, detuning, velocities)
        self.assertTrue(np.allclose(forces, -forces[::-1], rtol=1e-10, atol=1e-28))

    def test_diffusion_is_positive(self) -> None:
        rb_diffusion = diffusion_coefficient(
            RB87,
            DEFAULT_SIMULATION.cooling.saturation_parameter,
            DEFAULT_SIMULATION.cooling.detuning_rad_s(RB87),
        )
        yb_diffusion = diffusion_coefficient(
            YB171,
            DEFAULT_SIMULATION.cooling.saturation_parameter,
            DEFAULT_SIMULATION.cooling.detuning_rad_s(YB171),
        )
        self.assertGreater(rb_diffusion, 0.0)
        self.assertGreater(yb_diffusion, 0.0)
        self.assertGreater(rb_diffusion, yb_diffusion)

    def test_dipole_force_points_toward_center(self) -> None:
        beam_waist_m = DEFAULT_SIMULATION.tweezer.beam_waist_m
        self.assertLess(dipole_force(RB87, beam_waist_m, 0.3e-6), 0.0)
        self.assertGreater(dipole_force(RB87, beam_waist_m, -0.3e-6), 0.0)
        self.assertAlmostEqual(dipole_force(RB87, beam_waist_m, 0.0), 0.0)


class SimulationRegressionTests(unittest.TestCase):
    def test_zero_motion_keeps_atoms_trapped(self) -> None:
        config = DEFAULT_SIMULATION.with_overrides(trajectories=64, distance_m=0.0)
        rb = run_simulation(RB87, config)
        yb = run_simulation(YB171, config.with_overrides(random_seed=config.random_seed + 1))
        self.assertGreaterEqual(rb.survival_probability[-1], 0.95)
        self.assertGreaterEqual(yb.survival_probability[-1], 0.95)

    def test_shared_shuttle_heats_yb_and_reduces_survival(self) -> None:
        config = DEFAULT_SIMULATION.with_overrides(trajectories=128)
        rb = run_simulation(RB87, config)
        yb = run_simulation(YB171, config.with_overrides(random_seed=config.random_seed + 1))
        self.assertGreater(rb.survival_probability[-1], yb.survival_probability[-1])
        self.assertGreater(np.nanmax(yb.kinetic_temperature_k), np.nanmax(rb.kinetic_temperature_k))

    def test_survival_probability_is_monotonic(self) -> None:
        config = DEFAULT_SIMULATION.with_overrides(trajectories=64)
        yb = run_simulation(YB171, config)
        self.assertTrue(np.all(np.diff(yb.survival_probability) <= 1.0e-12))


class SpeedSweepTests(unittest.TestCase):
    def test_threshold_interpolates_between_bracketing_points(self) -> None:
        threshold = estimate_speed_threshold(
            np.array([0.1, 0.2, 0.3], dtype=float),
            np.array([1.0, 0.6, 0.2], dtype=float),
            0.5,
        )
        self.assertEqual(threshold.status, "bracketed")
        self.assertAlmostEqual(threshold.speed_limit_m_s, 0.225)
        self.assertAlmostEqual(threshold.lower_speed_m_s, 0.2)
        self.assertAlmostEqual(threshold.upper_speed_m_s, 0.3)

    def test_threshold_reports_above_sampled_range(self) -> None:
        threshold = estimate_speed_threshold(
            np.array([0.1, 0.2, 0.3], dtype=float),
            np.array([1.0, 0.9, 0.8], dtype=float),
            0.5,
        )
        self.assertEqual(threshold.status, "above_sampled_range")
        self.assertIsNone(threshold.speed_limit_m_s)
        self.assertAlmostEqual(threshold.lower_speed_m_s, 0.3)

    def test_speed_sweep_reports_distance_over_duration(self) -> None:
        durations = np.array([150.0e-6, 100.0e-6, 75.0e-6], dtype=float)
        config = DEFAULT_SIMULATION.with_overrides(trajectories=32, random_seed=2000)
        sweep = run_survival_speed_sweep(RB87, config, durations, target_survival_probability=0.5)
        expected_speeds = config.trajectory.distance_m / np.sort(durations)[::-1]
        self.assertTrue(np.allclose(sweep.average_speeds_m_s, np.sort(expected_speeds)))

    def test_yb_speed_limit_is_below_rb_speed_limit(self) -> None:
        durations = np.array([20.0e-6, 30.0e-6, 50.0e-6, 75.0e-6, 100.0e-6, 150.0e-6], dtype=float)
        config = DEFAULT_SIMULATION.with_overrides(trajectories=96)
        rb = run_survival_speed_sweep(RB87, config.with_overrides(random_seed=4000), durations)
        yb = run_survival_speed_sweep(YB171, config.with_overrides(random_seed=5000), durations)
        self.assertEqual(rb.threshold.status, "bracketed")
        self.assertEqual(yb.threshold.status, "bracketed")
        self.assertLess(yb.threshold.speed_limit_m_s, rb.threshold.speed_limit_m_s)

    def test_speed_scan_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "speed_scan"
            env = os.environ.copy()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atoms.speed_scan",
                    "--species",
                    "rb87",
                    "yb171",
                    "--durations-s",
                    "2e-5",
                    "3e-5",
                    "5e-5",
                    "7.5e-5",
                    "1e-4",
                    "1.5e-4",
                    "--trajectories",
                    "32",
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "survival_vs_speed.png").exists())
            self.assertTrue((output_dir / "speed_scan_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
