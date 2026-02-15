
import optuna
import numpy as np
import random
import os

from puff_simulation import get_objective, is_valid

"""
How to use optuna dashboard:

- https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html
- https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/011_journal_storage.html

Usage:

install optuna-dashboard

- pip install optuna-dashboard

Then call:

- optuna-dashboard <path_to_journalfile>

In your browser open:

- http://127.0.0.1:8080/ 

"""

# ==========================================
# 2. Objective Function (Wrapped for Optuna)
# ==========================================
def run_mock_puff_simulation(sensor_coords):
    """
    Mock of the expensive simulation call.
    In production, replace this with the actual 'gp_model.simulate()' logic.
    """
    # ... Real simulation code goes here ...
    # For demonstration, we return a random MAE that gets better 
    # if sensors are close to a "secret" target (e.g., 75, 75, 75).
    
    target = np.array([75.0, 75.0, 75.0])
    mae = 0.0
    for sensor in sensor_coords:
        dist = np.linalg.norm(sensor - target)
        mae += dist * 0.1  # Closer sensors = lower error
        
    return mae / len(sensor_coords)


from puff_simulation import get_objective

run_real_puff_simulation = get_objective()


def optuna_objective(trial):
    """
    The Optuna optimization target.
    """
    n_sensors = 2
    sensor_flat_list = []

    # --- A. Suggest Parameters ---
    for i in range(n_sensors):
        # We define the search space as 0-150 continuous
        x = trial.suggest_float(f"s{i}_x", -120, 120)
        y = trial.suggest_float(f"s{i}_y", -120, 120)
        z = trial.suggest_float(f"s{i}_z", -120, 120)
        
        # --- B. Apply Constraints ---
        # If the suggested point is invalid, we return a high penalty (Soft Constraint).
        # This teaches the TPE sampler to avoid these regions.
        if not is_valid(x, y, z):
            # Return a large error (Infinity or max possible error)
            # This effectively marks the trial as a failure without crashing
            raise optuna.TrialPruned("Invalid sensor location")
        
        sensor_flat_list.extend([x, y, z])

    # --- C. Run Simulation ---
    # Reshape for the physics engine
    sensor_coords = np.array(sensor_flat_list).reshape(-1, 3)
    
    try:
        # Call the actual objective function defined in previous turns
        mae = run_real_puff_simulation(sensor_coords) 
        if not np.isfinite(mae) or mae > 1E12:
            raise optuna.TrialPruned("Simulation returned invalid value")
        # Call a mock objective
        # mae = run_mock_puff_simulation(sensor_coords)
    except Exception as e:
        # If simulation fails (e.g., numerical instability), prune the trial
        print(f"Simulation failed: {e}")
        raise optuna.TrialPruned()

    return mae

# ==========================================
# 3. Main Optimization Loop with Journal
# ==========================================
if __name__ == "__main__":
    # Define the journal file path
    journal_file_path = "optuna_journal.log"
    
    # Create Journal Storage
    # This writes every trial to a file that the dashboard can read
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_file_path)
    )

    # Create the Study
    study_name = "sensor_optimization_v1"
    
    # Load existing study if it exists (resume), or create new
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",  # We want to minimize MAE
        load_if_exists=True
    )

    print(f"Starting optimization. Dashboard file: {os.path.abspath(journal_file_path)}")

    # Optimize!
    # n_trials=100 is just for demo; use more for convergence
    study.optimize(optuna_objective, n_trials=100, n_jobs=8)

    print("\nOptimization Complete.")
    print(f"Best MAE: {study.best_value}")
    print(f"Best Params: {study.best_params}")
        