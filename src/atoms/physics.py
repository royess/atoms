from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy import constants

from atoms.config import AtomicSpecies, SimulationConfig
from atoms.trajectory import build_time_grid, minimum_jerk_trajectory


def scattering_rate(gamma_rad_s: float, saturation_parameter: float, detuning_rad_s: np.ndarray | float) -> np.ndarray | float:
    denominator = 1.0 + saturation_parameter + 4.0 * (detuning_rad_s / gamma_rad_s) ** 2
    return 0.5 * gamma_rad_s * saturation_parameter / denominator


def cooling_force(
    species: AtomicSpecies,
    saturation_parameter: float,
    detuning_rad_s: float,
    velocity_m_s: np.ndarray | float,
) -> np.ndarray | float:
    doppler_shift = species.wave_number_m * velocity_m_s
    rate_forward = scattering_rate(species.gamma_rad_s, saturation_parameter, detuning_rad_s - doppler_shift)
    rate_backward = scattering_rate(species.gamma_rad_s, saturation_parameter, detuning_rad_s + doppler_shift)
    return constants.hbar * species.wave_number_m * (rate_forward - rate_backward)


def diffusion_coefficient(species: AtomicSpecies, saturation_parameter: float, detuning_rad_s: float) -> float:
    numerator = constants.hbar**2 * species.wave_number_m**2 * species.gamma_rad_s * saturation_parameter
    denominator = 2.0 * (1.0 + saturation_parameter + 4.0 * (detuning_rad_s / species.gamma_rad_s) ** 2)
    return numerator / denominator


def dipole_potential(species: AtomicSpecies, beam_waist_m: float, position_m: np.ndarray | float) -> np.ndarray | float:
    return -species.trap_depth_j * np.exp(-2.0 * (position_m / beam_waist_m) ** 2)


def dipole_force(species: AtomicSpecies, beam_waist_m: float, position_m: np.ndarray | float) -> np.ndarray | float:
    exponent = np.exp(-2.0 * (position_m / beam_waist_m) ** 2)
    return -(4.0 * species.trap_depth_j / (beam_waist_m**2)) * position_m * exponent


@dataclass
class SimulationResult:
    species: AtomicSpecies
    config: SimulationConfig
    time_s: np.ndarray
    trap_position_m: np.ndarray
    trap_velocity_m_s: np.ndarray
    trap_acceleration_m_s2: np.ndarray
    relative_position_m: np.ndarray
    momentum_kg_m_s: np.ndarray
    alive: np.ndarray
    kinetic_temperature_k: np.ndarray
    trapped_kinetic_temperature_k: np.ndarray

    @property
    def survival_probability(self) -> np.ndarray:
        return np.mean(self.alive, axis=0)

    @property
    def representative_position_m(self) -> np.ndarray:
        return self.relative_position_m[: self.config.phase_space_samples]

    @property
    def representative_momentum_kg_m_s(self) -> np.ndarray:
        return self.momentum_kg_m_s[: self.config.phase_space_samples]


@nb.njit(cache=True)
def _dipole_potential_numba(trap_depth_j: float, beam_waist_m: float, position_m: float) -> float:
    return -trap_depth_j * np.exp(-2.0 * (position_m / beam_waist_m) ** 2)


@nb.njit(cache=True)
def _dipole_force_numba(trap_depth_j: float, beam_waist_m: float, position_m: float) -> float:
    exponent = np.exp(-2.0 * (position_m / beam_waist_m) ** 2)
    return -(4.0 * trap_depth_j / (beam_waist_m**2)) * position_m * exponent


@nb.njit(cache=True)
def _scattering_rate_numba(gamma_rad_s: float, saturation_parameter: float, detuning_rad_s: float) -> float:
    denominator = 1.0 + saturation_parameter + 4.0 * (detuning_rad_s / gamma_rad_s) ** 2
    return 0.5 * gamma_rad_s * saturation_parameter / denominator


@nb.njit(cache=True)
def _cooling_force_numba(
    gamma_rad_s: float,
    wave_number_m: float,
    saturation_parameter: float,
    detuning_rad_s: float,
    velocity_m_s: float,
) -> float:
    doppler_shift = wave_number_m * velocity_m_s
    rate_forward = _scattering_rate_numba(gamma_rad_s, saturation_parameter, detuning_rad_s - doppler_shift)
    rate_backward = _scattering_rate_numba(gamma_rad_s, saturation_parameter, detuning_rad_s + doppler_shift)
    return constants.hbar * wave_number_m * (rate_forward - rate_backward)


@nb.njit(cache=True)
def _run_ensemble(
    trajectories: int,
    seed: int,
    mass_kg: float,
    wave_number_m: float,
    gamma_rad_s: float,
    trap_depth_j: float,
    beam_waist_m: float,
    saturation_parameter: float,
    detuning_rad_s: float,
    initial_temperature_k: float,
    time_step_s: float,
    acceleration_m_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    steps = acceleration_m_s2.shape[0]
    positions = np.empty((trajectories, steps))
    momenta = np.empty((trajectories, steps))
    alive = np.ones((trajectories, steps), dtype=np.bool_)
    spring_constant = 4.0 * trap_depth_j / (beam_waist_m**2)
    sigma_x = np.sqrt(constants.k * initial_temperature_k / spring_constant)
    sigma_p = np.sqrt(mass_kg * constants.k * initial_temperature_k)
    diffusion = (
        constants.hbar**2
        * wave_number_m**2
        * gamma_rad_s
        * saturation_parameter
        / (2.0 * (1.0 + saturation_parameter + 4.0 * (detuning_rad_s / gamma_rad_s) ** 2))
    )
    noise_prefactor = np.sqrt(2.0 * diffusion * time_step_s)

    for trajectory in range(trajectories):
        position = 0.0
        momentum = 0.0
        alive_state = True
        while True:
            position = sigma_x * np.random.normal()
            momentum = sigma_p * np.random.normal()
            energy = momentum * momentum / (2.0 * mass_kg) + _dipole_potential_numba(trap_depth_j, beam_waist_m, position)
            if energy < 0.0:
                break

        positions[trajectory, 0] = position
        momenta[trajectory, 0] = momentum
        alive[trajectory, 0] = True

        for step in range(1, steps):
            deterministic_force = _dipole_force_numba(trap_depth_j, beam_waist_m, position)
            deterministic_force += _cooling_force_numba(
                gamma_rad_s,
                wave_number_m,
                saturation_parameter,
                detuning_rad_s,
                momentum / mass_kg,
            )
            deterministic_force -= mass_kg * acceleration_m_s2[step - 1]

            momentum = momentum + deterministic_force * time_step_s + noise_prefactor * np.random.normal()
            position = position + (momentum / mass_kg) * time_step_s

            positions[trajectory, step] = position
            momenta[trajectory, step] = momentum

            energy = momentum * momentum / (2.0 * mass_kg) + _dipole_potential_numba(trap_depth_j, beam_waist_m, position)
            if alive_state and energy >= 0.0:
                alive_state = False
            alive[trajectory, step] = alive_state

    return positions, momenta, alive


def kinetic_temperature(momentum_kg_m_s: np.ndarray, mass_kg: float) -> np.ndarray:
    temperature = np.full(momentum_kg_m_s.shape[1], np.nan)
    for step in range(momentum_kg_m_s.shape[1]):
        if momentum_kg_m_s.shape[0] >= 2:
            temperature[step] = np.var(momentum_kg_m_s[:, step], ddof=1) / (mass_kg * constants.k)
    return temperature


def trapped_kinetic_temperature(momentum_kg_m_s: np.ndarray, alive: np.ndarray, mass_kg: float) -> np.ndarray:
    temperature = np.full(momentum_kg_m_s.shape[1], np.nan)
    for step in range(momentum_kg_m_s.shape[1]):
        active = alive[:, step]
        if np.count_nonzero(active) >= 2:
            temperature[step] = np.var(momentum_kg_m_s[active, step], ddof=1) / (mass_kg * constants.k)
    return temperature


def run_simulation(species: AtomicSpecies, config: SimulationConfig) -> SimulationResult:
    detuning_rad_s = config.cooling.detuning_rad_s(species)
    time_s = build_time_grid(config.trajectory.duration_s, config.time_step_s)
    trap_position_m, trap_velocity_m_s, trap_acceleration_m_s2 = minimum_jerk_trajectory(config.trajectory, time_s)
    positions, momenta, alive = _run_ensemble(
        trajectories=config.trajectories,
        seed=config.random_seed,
        mass_kg=species.mass_kg,
        wave_number_m=species.wave_number_m,
        gamma_rad_s=species.gamma_rad_s,
        trap_depth_j=species.trap_depth_j,
        beam_waist_m=config.tweezer.beam_waist_m,
        saturation_parameter=config.cooling.saturation_parameter,
        detuning_rad_s=detuning_rad_s,
        initial_temperature_k=config.initial_temperature_k,
        time_step_s=config.time_step_s,
        acceleration_m_s2=trap_acceleration_m_s2,
    )
    return SimulationResult(
        species=species,
        config=config,
        time_s=time_s,
        trap_position_m=trap_position_m,
        trap_velocity_m_s=trap_velocity_m_s,
        trap_acceleration_m_s2=trap_acceleration_m_s2,
        relative_position_m=positions,
        momentum_kg_m_s=momenta,
        alive=alive,
        kinetic_temperature_k=kinetic_temperature(momenta, species.mass_kg),
        trapped_kinetic_temperature_k=trapped_kinetic_temperature(momenta, alive, species.mass_kg),
    )
