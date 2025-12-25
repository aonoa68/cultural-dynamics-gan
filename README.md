# PlaceEmotion-GAN

PlaceEmotion-GAN is a Python package for simulating cultural dynamics and
emotional niche construction using a λ-scheduled generative adversarial
framework.

This repository accompanies the JOSS paper:
*“PlaceEmotion-GAN: A simulation-based GAN framework for modeling cultural
dynamics via λ-scheduling”*.

---

## Installation

```bash
pip install git+https://github.com/aonoa68/cultural-dynamics-gan.git
````

For development:

```bash
git clone https://github.com/aonoa68/cultural-dynamics-gan.git
cd cultural-dynamics-gan
pip install -e .
```

---

## Quick start (CLI)

Run a simple λ-scheduling simulation:

```bash
python -m placemotiongan.train --lambda-kind linear
```

Other scheduling options:

```bash
python -m placemotiongan.train --lambda-kind logistic
python -m placemotiongan.train --lambda-kind delayed
```

CSV logs and diagnostic plots are written to the specified output directory.

---

## Python API example

```python
import torch
from placemotiongan.train import simulate_training

logs, csv_path = simulate_training(
    total_steps=100,
    lambda_kind="linear",
)
```

---

## Repository structure

```text
placemotiongan/
├── model.py        # Generator / Discriminator definitions
├── scheduler.py    # λ-scheduling functions
├── losses.py       # Trade-off loss formulation
├── train.py        # Simulation-based training loop
analyses/
└── sensitivity.py  # λ-scheduling sensitivity analysis
```

---

## Reproducibility

All simulations are deterministic under a fixed random seed.
Sensitivity analyses comparing different λ-scheduling regimes are provided
in the `analyses/` directory.

---

## License

The software is released under the MIT License.
See `LICENSE` for details.

---

## Citation

If you use this software, please cite the accompanying paper:

```text
Onohara, A. (2025). PlaceEmotion-GAN: A simulation-based GAN framework for
modeling cultural dynamics via λ-scheduling. Journal of Open Source Software.
```

```

