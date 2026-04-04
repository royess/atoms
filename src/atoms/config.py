from __future__ import annotations

from dataclasses import dataclass, replace

from scipy import constants


def mw_per_cm2_to_w_per_m2(value_mw_cm2: float) -> float:
    return value_mw_cm2 * 10.0


@dataclass(frozen=True)
class AtomicSpecies:
    key: str
    label: str
    mass_u: float
    transition_wavelength_m: float
    gamma_rad_s: float
    saturation_intensity_w_m2: float
    trap_depth_k: float

    @property
    def mass_kg(self) -> float:
        return self.mass_u * constants.atomic_mass

    @property
    def wave_number_m(self) -> float:
        return 2.0 * constants.pi / self.transition_wavelength_m

    @property
    def trap_depth_j(self) -> float:
        return constants.k * self.trap_depth_k


@dataclass(frozen=True)
class TweezerConfig:
    beam_waist_m: float = 1.0e-6
    peak_intensity_w_m2: float = 2.0e10

    def effective_polarizability_si(self, species: AtomicSpecies) -> float:
        return 2.0 * constants.epsilon_0 * constants.c * species.trap_depth_j / self.peak_intensity_w_m2


@dataclass(frozen=True)
class CoolingConfig:
    saturation_parameter: float = 0.2
    detuning_in_gamma: float = -0.5

    def detuning_rad_s(self, species: AtomicSpecies) -> float:
        return self.detuning_in_gamma * species.gamma_rad_s


@dataclass(frozen=True)
class TrajectoryConfig:
    distance_m: float = 10.0e-6
    duration_s: float = 100.0e-6


@dataclass(frozen=True)
class SimulationConfig:
    trajectories: int = 1000
    time_step_s: float = 20.0e-9
    initial_temperature_k: float = 5.0e-6
    phase_space_samples: int = 24
    random_seed: int = 12345
    tweezer: TweezerConfig = TweezerConfig()
    cooling: CoolingConfig = CoolingConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()

    def with_overrides(
        self,
        *,
        trajectories: int | None = None,
        time_step_s: float | None = None,
        initial_temperature_k: float | None = None,
        phase_space_samples: int | None = None,
        random_seed: int | None = None,
        distance_m: float | None = None,
        duration_s: float | None = None,
        saturation_parameter: float | None = None,
        detuning_in_gamma: float | None = None,
        beam_waist_m: float | None = None,
        peak_intensity_w_m2: float | None = None,
    ) -> "SimulationConfig":
        next_config = self
        if trajectories is not None:
            next_config = replace(next_config, trajectories=trajectories)
        if time_step_s is not None:
            next_config = replace(next_config, time_step_s=time_step_s)
        if initial_temperature_k is not None:
            next_config = replace(next_config, initial_temperature_k=initial_temperature_k)
        if phase_space_samples is not None:
            next_config = replace(next_config, phase_space_samples=phase_space_samples)
        if random_seed is not None:
            next_config = replace(next_config, random_seed=random_seed)
        if distance_m is not None or duration_s is not None:
            next_config = replace(
                next_config,
                trajectory=replace(
                    next_config.trajectory,
                    distance_m=distance_m if distance_m is not None else next_config.trajectory.distance_m,
                    duration_s=duration_s if duration_s is not None else next_config.trajectory.duration_s,
                ),
            )
        if saturation_parameter is not None or detuning_in_gamma is not None:
            next_config = replace(
                next_config,
                cooling=replace(
                    next_config.cooling,
                    saturation_parameter=(
                        saturation_parameter
                        if saturation_parameter is not None
                        else next_config.cooling.saturation_parameter
                    ),
                    detuning_in_gamma=(
                        detuning_in_gamma
                        if detuning_in_gamma is not None
                        else next_config.cooling.detuning_in_gamma
                    ),
                ),
            )
        if beam_waist_m is not None or peak_intensity_w_m2 is not None:
            next_config = replace(
                next_config,
                tweezer=replace(
                    next_config.tweezer,
                    beam_waist_m=beam_waist_m if beam_waist_m is not None else next_config.tweezer.beam_waist_m,
                    peak_intensity_w_m2=(
                        peak_intensity_w_m2
                        if peak_intensity_w_m2 is not None
                        else next_config.tweezer.peak_intensity_w_m2
                    ),
                ),
            )
        return next_config


RB87 = AtomicSpecies(
    key="rb87",
    label="Rb-87",
    mass_u=86.909,
    transition_wavelength_m=780.24e-9,
    gamma_rad_s=2.0 * constants.pi * 6.06e6,
    saturation_intensity_w_m2=mw_per_cm2_to_w_per_m2(1.67),
    trap_depth_k=1.0e-3,
)

YB171 = AtomicSpecies(
    key="yb171",
    label="Yb-171",
    mass_u=170.936,
    transition_wavelength_m=555.8e-9,
    gamma_rad_s=2.0 * constants.pi * 182.0e3,
    saturation_intensity_w_m2=mw_per_cm2_to_w_per_m2(0.14),
    trap_depth_k=0.1e-3,
)

SPECIES_BY_KEY = {RB87.key: RB87, YB171.key: YB171}
DEFAULT_SIMULATION = SimulationConfig()
