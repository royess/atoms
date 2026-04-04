# AGENTS.md: Rigorous Semiclassical Atom Shuttling Simulator

## 1. System Role & Objective
**Role:** You are an expert Computational Physicist and Quantum Hardware Engineer.
**Objective:** Build a rigorous, first-principles Python/Julia simulation of neutral atom shuttling in optical tweezers. The goal is to quantitatively demonstrate the thermodynamic and kinetic differences between Alkali (Rb-87) and Alkaline-Earth-like (Yb-171) atoms during non-adiabatic transport.
**Constraint:** Do NOT use phenomenological damping constants (e.g., simple `F = -gamma * v`). You MUST calculate the dipole forces, scattering rates, and momentum diffusion directly from atomic physics parameters (AC Stark shift, spontaneous emission, Doppler cooling theory).

## 2. First-Principles Physics Engine

### 2.1 The Optical Tweezer (Dipole Potential)
Calculate the conservative potential derived from the AC Stark shift.
* Equation: U(r, z) = -(1 / (2 * epsilon_0 * c)) * Re(alpha) * I(r, z)
* I(r, z) is the Gaussian beam intensity profile.
* Force: F_dipole = -∇U(r, z).
* The trap center moves according to a predefined trajectory: x_trap(t).

### 2.2 Laser Cooling Force (Semiclassical S-matrix / Doppler)
Calculate the dissipative force from counter-propagating cooling lasers.
* For a two-level system approximation, the scattering rate is: 
  R_sc(v) = (Gamma / 2) * (s_0 / (1 + s_0 + (4 * (Delta - k*v)^2 / Gamma^2)))
* Where: 
  * Gamma = Natural linewidth (spontaneous emission rate)
  * Delta = Laser detuning
  * k = Wave vector of cooling light (2*pi / lambda)
  * s_0 = I / I_sat (Saturation parameter)
* Total Cooling Force (1D projection for simplicity): 
  F_cool = hbar * k * [R_sc(v_towards) - R_sc(v_away)]

### 2.3 Stochastic Heating (Momentum Diffusion)
Heating arises from the discrete, random nature of photon emission.
* Momentum diffusion coefficient: D_p = (hbar^2 * k^2 * Gamma / 2) * (s_0 / (1 + s_0 + 4*(Delta/Gamma)^2))
* The stochastic force term in the Langevin equation is: F_stoch = sqrt(2 * D_p) * W(t)
* W(t) is standard Gaussian white noise.

### 2.4 The Equation of Motion (SDE)
Solve the Stochastic Differential Equation using the Euler-Maruyama method:
dp = (F_dipole(x, t) + F_cool(v) + F_inertial) * dt + sqrt(2 * D_p) * dW
dx = (p / m) * dt
* Note: F_inertial = -m * a_trap(t) if simulating in the trap's moving reference frame.

## 3. Ground Truth Atomic Parameters

You MUST use these physical constants to initialize the atom objects.

**Rubidium-87 (Rb):**
* Mass: 86.909 u
* Cooling Transition: D2 line (780.24 nm)
* Linewidth (Gamma): 2*pi * 6.06 MHz
* Saturation Intensity (I_sat): 1.67 mW/cm^2
* Typical Trap Depth: 1.0 mK

**Ytterbium-171 (Yb):**
* Mass: 170.936 u
* Cooling Transition: Intercombination line ^1S_0 -> ^3P_1 (555.8 nm)
* Linewidth (Gamma): 2*pi * 182 kHz  <-- (CRITICAL: Notice how narrow this is compared to Rb)
* Saturation Intensity (I_sat): 0.14 mW/cm^2
* Typical Trap Depth: 0.1 mK (Constrained by magic wavelength requirements)

## 4. Implementation Requirements
1. **Tech Stack:** Python with NumPy, SciPy (for constants), and Numba (for SDE integration speed), OR Julia (DifferentialEquations.jl).
2. **Trajectory Generator:** Implement a Minimum-Jerk trajectory or a simple constant acceleration/deceleration profile for x_trap(t).
3. **Simulation Loop:** * Run N=1000 independent Monte Carlo trajectories for both Rb and Yb under the *exact same* movement profile (e.g., move 10 um in 100 us).
4. **Output & Visualization (Matplotlib):**
   * Plot 1: Phase Space Trajectories (Position vs. Momentum) in the trap frame.
   * Plot 2: Real-time kinetic temperature evolution: T_k(t) = Var(p) / (m * k_B).
   * Plot 3: Atom survival probability over time.

## 5. Expected Physical Outcome Validation
Before writing code, understand that the numerical results MUST show:
* Rb will easily stay cold due to its massive Gamma (high scattering rate = huge cooling power).
* Yb will rapidly heat up (T_k spikes) and leak out of the shallow trap because its narrow linewidth (182 kHz) provides insufficient damping against the non-adiabatic inertial forces of the movement.
