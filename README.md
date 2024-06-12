# HUDADE.jl

🚧 Disclaimer: This is still a prototype, in development and not officially registered! 🚧

## What is HUDADE.jl?
[*HUDADE.jl*](https://github.com/ThummeTo/HUDADE.jl) is a free-to-use software library for the Julia programming language with two goals:
1. It provides a common model interface (that is heavily motivated by the [functional mock-up interface](https://fmi-standard.org/)) of four functions - the HUDA-(O)DE - that is capable of describing a width variety of different simulation and machine learning models, technically mixed **H**ybrid, **U**niversal, **D**iscrete, **A**lgebraic and (**O**rdinary) **D**ifferential **E**quations.
2. It allows to combine multiple HUDA-(O)DEs in a very, very flexible way - much more than just parallel or sequential topologies. Even *learning* of arbitrary connections (gradient-based optimization) is supported.

HUDADE.jl provides the future foundation of FMI.jl and FMIFlux.jl, because any *FMU* and *neural FMU* can be expressed as HUDA-ODE.

## What is HUDADE.jl not?
[*HUDADE.jl*](https://github.com/ThummeTo/HUDADE.jl) should not be understood as modeling tool, there is *ModelingToolkit.jl* and *Causal.jl* for acausal and causal modeling. 
[*HUDADE.jl*](https://github.com/ThummeTo/HUDADE.jl) provides a model interface definition and inference implementation (`solve`), together with strategies to (learnable) combine multiple models in an interpretable fashion. 
Technically, this could be implemented some day as extension for existing modeling frameworks.

## What is a HUDA-ODE?
A HUDA-ODE is a common model interface of five functions (four non-trivial), that is capable of describing a width variety of different simulation and machine learning models, technically mixed **H**ybrid, **U**niversal, **D**iscrete, **A**lgebraic and (**O**rdinary) **D**ifferential **E**quations.
The system of equations is defined as follows:
```math
\begin{bmatrix}
	\dot{\vec{x}}_c(t) \\ 
	\dot{\vec{x}}_d(t) \\ 
	\vec{y}(t) \\
	\vec{z}(t) \\
	\vec{x}(t^+)
\end{bmatrix} = 
\begin{bmatrix}
	\vec{f}(\vec{x}_c(t), \vec{x}_d(t), \vec{u}(t), \vec{p}, t) \\
	\vec{0} \\
	\vec{g}(\vec{x}_c(t), \vec{x}_d(t), \vec{u}(t), \vec{p}, t) \\
	\vec{c}(\vec{x}_c(t), \vec{x}_d(t), \vec{u}(t), \vec{p}, t) \\
	\vec{a}(\vec{x}_c(t^-), \vec{x}_d(t^-), \vec{u}(t^-), \vec{p}, t^-)
\end{bmatrix}
```
with system continuous state $\vec{x}_c$, discrete state $\vec{x}_d$, inputs $\vec{u}$, parameters $\vec{p}$ and time $t$. A more detailed introduction can be found in the linked paper.

## Limitations
For now, the current implementation is restricted to ODEs, so PDEs or DAEs are not supported out of the box.
But this is planned for the future.

## Achknowledgement
From a software point of view, this package provides a common interface for differentiable simulation and machine learning models and therefore heavily relies, besides many many others, on the packages [*DifferentialEquations.jl*](https://github.com/SciML/DifferentialEquations.jl), [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) and [DiffEqCallbacks.jl](https://github.com/SciML/DiffEqCallbacks.jl).
[*HUDADE.jl*](https://github.com/ThummeTo/HUDADE.jl) provides a versatile model interface on top of them.

## Further details and citation:
Tobias Thummerer and Lars Mikelsons. 2024. **Learnable & Interpretable Model Combination in Dynamic Systems Modeling.** ArXiv (Preprint).
