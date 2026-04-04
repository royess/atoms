# Semiclassical Atom Shuttling Simulator

This repo contains a first-principles 1D semiclassical shuttle model for `Rb-87` and `Yb-171` in a moving optical tweezer. The dynamics are integrated in the trap frame with:

- AC-Stark dipole trapping from a depth-calibrated Gaussian tweezer potential
- Counter-propagating Doppler cooling from the prompt's two-level scattering model
- Momentum diffusion from spontaneous-emission recoil
- Euler-Maruyama Monte Carlo integration over many trajectories

The default transport profile is a `10 um` minimum-jerk move completed in `100 us` for `1000` trajectories per species.

## Environment

Create the local Python environment with the `uv` executable from the conda install:

```bash
/home/yuxuan/anaconda3/bin/uv venv .venv
/home/yuxuan/anaconda3/bin/uv pip install --python .venv/bin/python numpy scipy numba matplotlib
/home/yuxuan/anaconda3/bin/uv pip install --python .venv/bin/python -e .
```

## Run

Run the default two-species comparison:

```bash
PYTHONPATH=src .venv/bin/python -m atoms.simulate
```

Write outputs to a custom directory:

```bash
PYTHONPATH=src .venv/bin/python -m atoms.simulate --output-dir outputs/run_001
```

Useful overrides:

```bash
PYTHONPATH=src .venv/bin/python -m atoms.simulate \
  --trajectories 2000 \
  --time-step-s 1e-8 \
  --distance-m 1e-5 \
  --duration-s 1e-4 \
  --saturation-parameter 0.2 \
  --detuning-in-gamma -0.5
```

The CLI writes:

- `phase_space.png`
- `kinetic_temperature.png`
- `survival_probability.png`
- `summary.json`

## Notes

- The atomic constants match `AGENTS.md`.
- The linewidths are used exactly as given there, including the explicit `2*pi` factors.
- The tweezer force is implemented through `U(x) = -(1/(2 eps0 c)) Re(alpha_eff) I(x)` with `alpha_eff` back-computed from the requested trap depth and the configured peak intensity.
- Survival is defined by the 1D classical bound-state condition `p^2/(2m) + U(x) < 0`.
- `kinetic_temperature.png` reports the full-ensemble momentum variance. `summary.json` also records the final trapped-only kinetic temperature when survivors remain.

## Tests

Run the unit and regression tests with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
