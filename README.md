# SymBoltz Implementation of Pure Momentum Transfer involving the Dark Sector

Author: **David Figueruelo**

This repository contains an **implementation in SymBoltz** of a **pure momentum transfer cosmological model** at the level of linear cosmological perturbations.

The code includes three possible momentum-transfer interaction channels:

- **Dark Energy – Dark Matter** with coupling parameter **α**
- **Dark Energy – Baryons** with coupling parameter **β**
- **Dark Matter – Baryons** with coupling parameter **γ**
	@@ -14,7 +14,7 @@ All three interactions are implemented within a single symbolic SymBoltz model,

The goal of this repository is to provide a flexible and transparent **SymBoltz implementation** of these interacting cosmologies, suitable for computing both:

- the **matter power spectrum** \( P(k) \)
- the **CMB angular power spectra** (for example **TT**, **EE**, and **TE**)

The implementation is based on the same physical momentum-transfer framework developed in the literature cited below, extended here to a unified SymBoltz implementation of the full **pure momentum** model.
	@@ -25,7 +25,7 @@ The implementation is based on the same physical momentum-transfer framework dev

# Running the Code

Run Julia with multithreading enabled

```
julia -tauto
	@@ -41,12 +41,12 @@ include("pure_momentum_symboltz.jl")

Only change **USER CONFIG** of the .jl file. There you can choose the values of the coupling parameter for each pure momentum transfer, the value of the other cosmological parameters and where to place the output. In particular:
- **coupling_sets**            -> choose the value of   α , β and γ
- **run_mode**                 -> choose if you want the matter power spectrum (:pk) or the CMB (:cmb), or both (:both)
- **cmb_modes**                -> choose which CMB you want (:TT, :EE, :TE)
- **Cosmology params**         -> choose the value of the common cosmological parameters
- **P(k) grid and CMB grid**   -> choose k and l grid for :pk and/or :cmb

**Other sections of the code deal with the cosmology, change at your own risk.**

---

	@@ -90,7 +90,7 @@ The repository is organized around a unified symbolic implementation of the **pu

The main Julia scripts allow the computation of the observables:

- **matter power spectrum \( P(k) \)**
- **CMB angular power spectra**
  - **TT**
  - **EE**
	@@ -104,7 +104,7 @@ Depending on the script configuration, the user can choose:

The code is designed so that the user can easily run:

- one single cosmology
- several coupling configurations
- one or several observables in a single execution

	@@ -154,7 +154,7 @@ where the coupling rate Γβ is proportional to the parameter **β**.

## 3. Dark Matter – Baryons

This code implements an **elastic scattering (momentum transfer) interaction between Dark MAtter and Baryons**, modifying the **velocity (θ) equations** of DM and B at the level of **linear cosmological perturbations**.

The interaction modifies the perturbation equations through terms of the form

Γγ (θ_B − θ_DM)
in the Baryons velocity equation, and a corresponding term in the Dark Matter velocity equation
Γγ Rγ (θ_B − θ_DM)
where the coupling rate Γγ is proportional to the parameter **γ**.
---
# Model Parameters
The interaction strengths are controlled by the coupling parameters
α, β, γ
where:
- **α** controls momentum transfer between **Dark Energy and Dark Matter**
- **β** controls momentum transfer between **Dark Energy and Baryons**
- **γ** controls momentum transfer between **Dark Matter and Baryons**
These parameters can be set independently.
---
# Validation Against CLASS
The SymBoltz implementation has been directly compared with the CLASS implementation of the same model.
The resulting cosmological observables agree at the **sub-percent level**, confirming the correctness of the symbolic implementation.
![Pk comparison between CLASS and SymBoltz](plots/Pk_CLASS_vs_SymBoltz_modelo.png)
---
# Observables
The repository can compute the following cosmological observables with SymBoltz.
## Matter Power Spectrum
- **P(k)**
## CMB Angular Power Spectra
- **TT**
- **EE**
- **TE**
The user can choose which observable(s) to compute directly from the Julia script configuration.
---
# Requirements
Julia ≥ 1.9
Required packages:
- SymBoltz
- Plots
- Unitful
- UnitfulAstro
Example installation:
```
using Pkg
Pkg.add("SymBoltz")
Pkg.add("Plots")
Pkg.add("Unitful")
Pkg.add("UnitfulAstro")
```
# Citations
If you use this code in scientific work, please cite at least the following references:
### Momentum transfer model DE-DM
```
@article{Figueruelo:2021elm,
  author = "Figueruelo, David and others",
  title = "{J-PAS: Forecasts for dark matter - dark energy elastic couplings}",
  eprint = "2103.01571",
  archivePrefix = "arXiv",
  primaryClass = "astro-ph.CO",
  doi = "10.1088/1475-7516/2021/07/022",
  journal = "JCAP",
  volume = "07",
  pages = "022",
  year = "2021"
}
```
### Momentum transfer model DE-B
```
@article{BeltranJimenez:2020iyx,
    author = "Beltr{\'a}n Jim{\'e}nez, Jose and Bettoni, Dario and Figueruelo, David and Teppa Pannia, Florencia A.",
    title = "{On cosmological signatures of baryons-dark energy elastic couplings}",
    eprint = "2004.14661",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2020/08/020",
    journal = "JCAP",
    volume = "08",
    pages = "020",
    year = "2020"
}
```
### Momentum transfer model DM-B (to be announced)
### SymBoltz
```
@article{SymBoltz,
  title = {{SymBoltz.jl}: a symbolic-numeric, approximation-free and differentiable linear {Einstein-Boltzmann} solver},
  author = {Herman Sletmoen},
  year = {2025},
  journal = {arXiv},
  eprint = {2509.24740},
  archiveprefix = {arXiv},
  primaryclass = {astro-ph.CO},
  doi = {10.48550/arXiv.2509.24740},
  url = {http://arxiv.org/abs/2509.24740}
}
```
---
# Contact
**David Figueruelo**  
david.figueruelo@ehu.eus
Universidad del País Vasco / Euskal Herriko Unibertsitatea (UPV/EHU)  
Investigador especialización doctores (2024)
