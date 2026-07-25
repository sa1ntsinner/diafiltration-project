# Model reference

A compact, self-contained statement of the model, its parameters and the
properties used by the controllers.  Derivations and discussion are in
[`REPORT.md`](REPORT.md); the code is `src/dfp/model.py`.

## States, input, outputs

| symbol | meaning | unit |
|---|---|---|
| `x₁ = V` | retentate volume | m³ |
| `x₂ = M_L` | lactose hold-up | mol |
| `x₃ = M_P` | protein hold-up | mol |
| `u = d/p` | solvent-to-permeate ratio (manipulated) | – |
| `c_P = M_P/V`, `c_L = M_L/V` | concentrations | mol m⁻³ |
| `p = kA ln(c_g/c_P)` | permeate volumetric flow | m³ s⁻¹ |

## Equations

```
dV/dt   = (u − 1) · p
dM_L/dt = − r_L · c_L · p
dM_P/dt = − r_P · c_P · p          (≡ 0 for complete protein retention)

r_L = α / [ 1 + (α − 1) · exp( p / (k_{M,L} A) ) ]        (Eq. 2)
r_P = β / [ 1 + (β − 1) · exp( p / (k_{M,P} A) ) ]        (Eq. 6)
```

`r_•` is the permeate/retentate concentration ratio.  For `γ ≥ 1` and `p > 0` it
lies in `(0, 1]`, i.e. the membrane can never enrich the permeate above the
retentate.  For `γ < 1` the denominator turns negative and the expression is
unphysical, which is why complete protein retention is modelled by a build flag
(`dM_P/dt ≡ 0`) rather than by `β = 0`.

## Parameters (`dfp.config.ProcessParams`)

| symbol | value | unit | note |
|---|---:|---|---|
| `k` | 4.79 × 10⁻⁶ | m s⁻¹ | task sheet prints `m s⁻²`; see README |
| `A` | 1.0 | m² | |
| `c_g` | 319 | mol m⁻³ | gel-layer concentration |
| `k_{M,L}` | 1.6 × 10⁻⁵ | m s⁻¹ | uncertain: 0.25 – 1.1 × in the studies |
| `α` | 1.3 | – | lactose partition function |
| `β` | 1.3 | – | protein partition function (leakage scenario) |
| `k_{M,P}` | 1 × 10⁻⁶ | m s⁻¹ | leakage scenario; negligible below 2.8 × 10⁻⁷ |
| `V₀`, `c_{P,0}`, `c_{L,0}` | 0.10, 10, 150 | m³, mol m⁻³ | initial charge |
| `c_{P,f}`, `c_{L,f}`, `c_{L,max}` | 100, 15, 570 | mol m⁻³ | specifications |
| `Δt`, `t_max` | 600, 21600 | s | control interval, horizon |

Derived: `M_{P,0} = 1 mol`, `M_{L,0} = 15 mol`, `V_f = M_{P,0}/c_{P,f} = 10 L`,
`V_gel = M_{P,0}/c_g = 3.13 L`.

## Properties used by the controllers

1. **Monotone volume.** `u ≤ 1 ⇒ dV/dt ≤ 0`, so `c_P` never decreases on the
   nominal plant and can never overshoot its (equality) specification unless the
   controller lets it.
2. **Non-linearity.** Logarithmic flux, exponential partition ratio and the
   bilinear inflow `u·p(V)`; no coordinate change removes all three.
3. **Linearity in `σ = 1/(1−u)`** after the change of independent variable
   `t → c_P`.  This is what makes the time-optimal problem a linear program and
   yields the bang-bang solution.
4. **Washing efficiency grows with `c_P`.** `r_L`: 0.704 at `c_P = 10` →
   0.913 at `c_P = 100`; specific washing rate `p c_P r_L / M_P` rises 4.3 ×.
5. **Leakage inverts the washing arc.**
   `dc_P/dt = (p c_P/V)(1 − u − r_P)`, so with `r_P > 0` the constant-volume
   arc `u = 1` *reduces* `c_P`.

## Uncertainty sets

`dfp.config.KM_L_UNCERTAINTY` — the four realisations of the task sheet,
`k_{M,L} ∈ {0.25, 0.5, 0.75, 1.0} × k_{M,L}^{nom}`, used as the scenario tree of
the multi-stage NMPC.

`dfp.experiments.montecarlo.DEFAULT_RANGES` — the Monte-Carlo box:
`k_{M,L} ∈ [0.25, 1.10]`, `k ∈ [0.80, 1.20]`, `α ∈ [0.95, 1.15]`,
`c_g ∈ [0.95, 1.05]`, all multiplicative and uniform.
