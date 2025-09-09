# SynModel: A Framework for Reproducible Cortical Microcircuit Modeling with Spiking Neural Networks
[![arXiv](https://img.shields.io/badge/arXiv-2409.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2409.XXXXX)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

We introduce SynModel, a computational framework relying on Brian 2. SynModel provides a reproducible way to construct, simulate, and validate cortical microcircuit models against experimental recordings [Submitted to NeurIPS 2025 Workshop: Data on the Brain & Mind]

## Usage Guide

### 1. Environment Setup

```bash
python3 -m venv venv # Linux/Mac
python -m venv venv # Windows

source venv/bin/activate  # Linux/Mac
venv\Scripts\activate # Windows

pip install -r requirements.txt
```

**Important Note on Data Download:** The first time you run `scripts/run_simulation.py`, the AllenSDK will automatically download several large electrophysiology data files (2GB).

### 2. Running the Simulations

The framework includes five scripts, with graphical outputs to `figures/` numerical outputs to `results/`.

#### 2a. Main Data-Driven Simulation

Simulation of a cortical microcircuit and compares its output statistics against in-vivo data from the Allen Brain Observatory.

```bash
python scripts/run_simulation.py --duration 5
```
*   `--duration`: Sets the simulation time in seconds. Defaults to 5.

#### 2b. Multi-Layer STDP Simulation

Demonstrates a four-layer feedforward network where synaptic weights evolve according to Spike-Timing-Dependent Plasticity (STDP), leading to self-organized dynamics.

```bash
python scripts/simple_lif_simulation.py
```

#### 2c. 1-Back Working Memory Task

A simple cognitive task to show how the framework can be used to model working memory.

```bash
python scripts/one_back_task_simulation.py
```

#### 2d. Neuromodulation Demo

Demonstrates the effects of simulated dopamine (on plasticity) and acetylcholine (on excitability) in a simple two-neuron circuit.

```bash
python scripts/neuromodulation_demo.py
```

#### 2e. Cognitive Signal Analysis

A simulation and performs advanced analysis on the simulated Local Field Potential (LFP), including calculating Phase-Amplitude Coupling (PAC) and coherence.

```bash
python scripts/run_cognitive_analysis.py
```

#### 2f. AdEx Neuron Demo

This script runs a simulation of a single population of Adaptive Exponential (AdEx) neurons to demonstrate their characteristic spike-frequency adaptation.

```bash
python scripts/adex_simulation_demo.py
```

#### 2g. Explore Allen Institute Data

This script provides a way to directly visualize the Allen Brain Observatory data used for validation. It plots key metrics like the distribution of firing rates, a raster for a single neuron, and a sample of the LFP.

```bash
python scripts/explore_allen_data.py
```

### 3. Reproducing the Paper's Key Results

Once all simulations have been run, execute the results analysis script:

```bash
python scripts/results.py
```

## Citation

```bibtex
@misc{Anon2025SynModel,
  title={SynModel: Biologically Plausible Cortical Microcircuit Modeling with Spiking Neural Networks},
  author={Anon},
  year={2025},
  eprint={2409.XXXXX},
  archivePrefix={arXiv},
  primaryClass={q-bio.NC},
  note={TBD}
}
```

## Contact

For any inquiries, please contact ANON or the authors at ANON.
