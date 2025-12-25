---
title: "PlaceEmotion-GAN: A simulation-based GAN framework for modeling cultural dynamics via λ-scheduling"
tags:
  - Python
  - cultural evolution
  - generative adversarial networks
  - emotion modeling
  - simulation
authors:
  - name: Ayaka Onohara
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Hokusei Gakuen University
    index: 1
date: 2025-01-01
bibliography: references.bib
---

# Summary

PlaceEmotion-GAN is a Python software package that implements a simulation-based
generative adversarial framework for studying cultural dynamics and emotional
niche construction.
Rather than focusing on large-scale neural network training, the package
emphasizes the explicit modeling of trade-offs between adversarial pressure and
the emergence of structured cultural-emotional features through a controllable
λ-scheduling mechanism.

The software provides modular components for (i) generator and discriminator
models, (ii) time-dependent λ-scheduling functions, (iii) theory-driven loss
functions, and (iv) a reproducible training simulation loop.
Together, these components allow researchers to systematically explore how
different temporal schedules of optimization pressure affect qualitative
learning dynamics.

# Statement of need

Generative adversarial networks (GANs) have been widely adopted as flexible
models of learning and adaptation, but their use in cultural and social modeling
often remains opaque due to tightly coupled implementations and implicit
optimization dynamics.
In particular, many existing implementations obscure the temporal trade-off
between exploration and exploitation that is central to theories of cultural
evolution and niche construction.

PlaceEmotion-GAN addresses this gap by providing a lightweight, modular, and
theory-aligned software framework in which the temporal structure of learning is
explicitly parameterized.
The software is designed for researchers in cultural evolution, computational
social science, and emotion modeling who require transparent and reproducible
tools for investigating how learning dynamics depend on time-dependent
constraints rather than raw model capacity.

# Software description

## Architecture

The package is organized into four core modules:

- `model.py`: Defines minimal generator and discriminator neural networks that
  map latent variables to cultural-emotional feature representations.
- `scheduler.py`: Implements λ-scheduling functions (linear, logistic, and
  delayed) that control the relative weight of competing objectives over time.
- `losses.py`: Provides a theory-driven trade-off loss function that combines
  adversarial loss and feature-based reward under the λ schedule.
- `train.py`: Integrates the above components into a reproducible,
  simulation-based training loop with command-line support.

In addition, the `analyses/` directory contains sensitivity analysis scripts
that systematically compare different λ-scheduling regimes under controlled
random seeds.

## λ-scheduling

A central contribution of the software is the explicit implementation of
λ-scheduling as a first-class component.
The scheduler returns a value λ(t) ∈ [0, 1] at each training step, controlling
the balance between adversarial pressure and feature reward.
Three scheduling shapes are provided:

- Linear: monotonic increase toward λ = 1
- Logistic: sigmoidal transition with adjustable sharpness
- Delayed: an initial plateau followed by a rapid transition

These schedules enable controlled experiments on how the timing of structural
constraints influences learning dynamics.

## Loss formulation

The total loss is defined as:

\[
L(t) = (1 - \lambda(t)) \cdot L_{\text{adv}} - \lambda(t) \cdot \beta \cdot R
\]

where \(L_{\text{adv}}\) is the adversarial loss, \(R\) is a feature-based reward,
and \(\beta\) controls the relative scale of the reward term.
This formulation directly reflects theoretical assumptions about trade-offs
between exploratory learning and structured cultural adaptation.

## Reproducibility and usage

The training loop is implemented as a simulation rather than a full-scale
optimization pipeline, allowing deterministic reproduction of qualitative
dynamics under fixed random seeds.
All core functionality can be accessed via both a Python API and a command-line
interface.
CSV logs and diagnostic plots are automatically generated to facilitate further
analysis.

# Acknowledgements

The author thanks colleagues in cultural evolution and computational social
science for discussions that motivated the explicit treatment of temporal
trade-offs in learning dynamics.

# References
