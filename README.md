# MuJoCo Human Standing Optimization

## Introduction

This repository provides a simulation and optimization framework for human standing control using MuJoCo.

The framework includes:

- Human standing simulation using a musculoskeletal model
- Feedback (PD-based) control
- Feedforward muscle activation optimization
- Multi-objective optimization (e.g., NSGA-II and COMO-CMA-ES)
- Rendering of optimized motions
- Visualization of optimization results

---

## Requirements

The project has been tested on:

- Windows 11
- Python 3.10 or later
- Miniconda / Anaconda
- Git

Please install the following software before proceeding:

- Git: https://git-scm.com/install/windows
- Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install/overview
- VScode: https://code.visualstudio.com/download?_exp_download=fb315fc982

---

## Installation

Clone this repository.

### HTTPS

```bash
git clone https://github.com/shinshu-chibalab/myo.git
```

### SSH

```bash
git clone git@github.com:shinshu-chibalab/myo.git
```

Move into the project directory.

```bash
cd myo
```

---

## Create the Conda Environment

Create a Conda environment using the provided `environment.yml`.

```bash
conda env create -f environment.yaml -n <env-name>
```

You may choose any environment name.

---

## Activate the Environment

```bash
conda activate <env-name>
```

Replace `<env-name>` with the name you specified when creating the environment.

---

## Project Structure

```
myo/
├── controller/         # Controllers
├── evaluator/          # Objective functions
├── myo_sim/            # MuJoCo models
├── optim/              # Optimization algorithms
├── render/             # Visualization and rendering utilities
├── results/            # Optimization results
├── simulation/         # Simulation workers
├── utils/              # Utility functions
├── standing_*.py       # Main optimization scripts
├── x0_*.py             # Initial parameter definitions
├── test_scripts.py     # Test scripts
├── cos_sim.py          # Cosine similarity and norm calculation
├── camera.py           # MuJoCo model viewer
└── environment.yml     # Conda environment file
```

---

## Running an Optimization

Example:

```bash
python standing_nsga2.py
```

or

```bash
python standing_como_cma_es.py
```

---

## Output

Optimization results are automatically saved in

```
results/
```

including:

- Pareto front
- Optimization history
- Rendered videos
- Cost history
- Optimized parameters
- Muscle activation plots

---

## Rendering

Optimized motions are rendered automatically after the optimization is completed.

Generated videos are stored in

```
results/<model_name>/videos
```

---

## Notes

- MuJoCo must be installed correctly before running the project.
- Musculoskeletal models are located in `myo_sim/`.
- To modify the simulation process, edit the files in `simulation/`.
- To implement or modify optimization algorithms, edit the files in `optim/`.
- To modify the muscle controller, edit the files in `controller/`.
- To change the objective functions, edit the files in `evaluator/`.

---

## License

This project is released under the MIT License unless otherwise specified.
