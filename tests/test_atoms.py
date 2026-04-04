from __future__ import annotations

import unittest

import numpy as np
from scipy import constants

from atoms.config import DEFAULT_SIMULATION, RB87, YB171
from atoms.physics import cooling_force, diffusion_coefficient, dipole_force, run_simulation
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


if __name__ == "__main__":
    unittest.main()
