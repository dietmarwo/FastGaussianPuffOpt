# Sensor Optimization: FastGaussianPuff + fcmaes + Optuna Dashboard
image::logo.gif[]

This project demonstrates how to solve a physics-based **Inverse Design Problem**—optimizing the placement of methane sensors—using two different optimization strategies. 

It highlights a unique "best of both worlds" approach: utilizing the raw speed and parallel efficiency of **fcmaes (BiteOpt)** while maintaining the user-friendly visualization of **Optuna Dashboard**.

## 🎯 The Objective

**Goal:** Find the optimal $(x, y, z)$ coordinates for a set of methane sensors to maximize detection accuracy (minimize Mean Absolute Error against a ground truth).

**The Challenge:**
1.  **Expensive Physics:** The objective function requires running a **Gaussian Puff** dispersion simulation (advection-diffusion) over time.
2.  **Complex Constraints:** Sensors must be placed in valid "free space" within a voxel grid (avoiding buildings/obstacles).
3.  **Visualization:** We need to see the optimization progress in real-time to diagnose convergence issues.

## 🚀 Key Features

### 1. The Physics Engine (`FastGaussianPuff`)
Uses a custom C++/Python implementation of the Gaussian Puff model [Jia et al., 2023] to simulate methane plumes. This is significantly faster than LES (Large Eddy Simulations) but still computationally intensive enough to require efficient optimization.

### 2. The "Bridge" (`journal.py`)
This is the core innovation of this repository. It is a custom wrapper that allows **fcmaes**—a high-performance, derivative-free C++ optimizer—to write its results directly to an **Optuna Journal** file.
* **Why?** fcmaes is faster and handles parallelism better than standard Python loops, but lacks a native GUI.
* **The Fix:** `journal.py` intercepts the optimization results and writes them to `optuna_journal.log` in a format `optuna-dashboard` understands.
* **Smart Pruning:** It automatically detects `np.inf` (constraint violations) and `NaN` (simulation crashes) and logs them as **PRUNED** trials. This prevents the dashboard charts from scaling to infinity, keeping your valid data readable.

### 3. Optimizer Comparison
* **Standard:** `optuna` (TPE Sampler) - Good for general hyperparameters, but slower on mathematical functions.
* **Accelerated:** `fcmaes` (BiteOpt/CMA-ES) - Excellent for continuous optimization problems, robust against local minima, and utilizes native C++ multi-threading.

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `puff_simulation.py` | **The Simulation Engine.** Sets up the wind data, source location, and ground truth. Defines `is_valid(x,y,z)` for geometric constraints. |
| `journal.py` | **The Bridge.** A custom logging wrapper that translates `fcmaes` events into Optuna Journal format. Handles `inf` pruning. |
| `opt_optuna.py` | **Baseline Optimization.** Runs the problem using standard Optuna TPE. |
| `opt_fcmaes_j.py` | **Accelerated Optimization.** Runs the problem using `fcmaes` wrapped in `journal.py` for dashboard compatibility. |
| `GaussianPuff.py` | Python wrapper for the C++ physics engine. |
| `CGaussianPuff.cpp` | The core C++ physics implementation (must be compiled). |


### Optuna Dashboard / Optuna Optimizer
image::optuna.png[]

### Optuna Dashboard / fcmaes Optimizer
image::fcmaes.png[]

## 🛠️ Installation

### 1. Prerequisites
To run the `opt_` files, you must install the following core packages:

* **FastGaussianPuff**: The physics engine is mandatory. Clone and install it from the Hammerling Research Group repository:
    * [https://github.com/Hammerling-Research-Group/FastGaussianPuff](https://github.com/Hammerling-Research-Group/FastGaussianPuff)
* **Optuna**: The optimization framework used for the baseline and journal storage.
    * [https://optuna.org/](https://optuna.org/)
* **Optuna Dashboard**: The real-time visualization tool.
    * [https://github.com/optuna/optuna-dashboard](https://github.com/optuna/optuna-dashboard)
* **fcmaes**: The high-performance optimization library.
    * [https://github.com/dietmarwo/fast-cma-es](https://github.com/dietmarwo/fast-cma-es)

### 2. Install Python Dependencies
Once the prerequisites above are addressed (specifically `FastGaussianPuff`, which requires C++ compilation), install the remaining Python dependencies:

```bash
pip install numpy pandas scipy optuna optuna-dashboard loguru fcmaes
```

## 🏃 Usage

### 1. Run the Standard Approach (Optuna)
Good for establishing a baseline.
```bash
python opt_optuna.py
```

### 2. Run the Accelerated Approach (fcmaes + Journal)
Uses the `Bite_cpp` algorithm with parallel retries. This script uses `journal.py` to log results to `fcmaes_journal.log` (or `optuna_journal.log` depending on config).
```bash
python opt_fcmaes_j.py
```

### 3. Visualize in Real-Time
Launch the dashboard to watch the optimization. The custom wrapper ensures that invalid sensor positions (which return `infinity`) are marked as "Pruned" so they don't break your charts.

```bash
# If running opt_optuna.py
optuna-dashboard optuna_journal.log

# If running opt_fcmaes_j.py
optuna-dashboard fcmaes_journal.log
```

Open your browser to `http://127.0.0.1:8080/`.

## 📊 What to Look For
* **Timeline:** See how `fcmaes` runs multiple workers in parallel (green bars).
* **History Plot:** Observe the convergence. Notice that failed trials (invalid positions) do not distort the Y-axis range because they are handled as `PRUNED` states.
* **Parallel Coordinate:** Visualize how the $(x, y, z)$ variables converge to specific regions in the domain.

## 📚 References
* **Gaussian Puff Model**: [ChemRxiv Preprint (Jia et al., 2023)](https://doi.org/10.26434/chemrxiv-2023-hc95q-v2).
* **fcmaes**: [Fast CMA-ES implementation](https://github.com/dietmarwo/fast-cma-es).
