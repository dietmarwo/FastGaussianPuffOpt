import optuna
import numpy as np
import random
import os

# Import the objective factory and validation logic from our simulation file
from puff_simulation import get_objective, is_valid

# Initialize the simulation function.
# This prepares the ground truth data once, so we don't re-compute it every trial.
run_real_puff_simulation = get_objective()

def optuna_objective(trial):
    """
    The main callback function for the Optuna optimizer.
    Optuna calls this function repeatedly with different parameter suggestions.
    """
    
    # Configuration: How many sensors are we optimizing?
    n_sensors = 2
    sensor_flat_list = []

    # --- A. Suggest Parameters (The Search Space) ---
    for i in range(n_sensors):
        # Suggest float coordinates for each sensor (x, y, z).
        # We search in the range [-120, 120].
        # These correspond to the spatial domain defined in puff_simulation.py.
        x = trial.suggest_float(f"s{i}_x", -120, 120)
        y = trial.suggest_float(f"s{i}_y", -120, 120)
        z = trial.suggest_float(f"s{i}_z", -120, 120)
        
        # --- B. Apply Constraints (Position Filtering) ---
        # Before running the expensive simulation, check if the position is valid.
        # 'is_valid' checks against the boolean grid (avoiding obstacles).
        if not is_valid(x, y, z):
            # Soft Constraint technique:
            # Instead of crashing, we return 'np.inf' (Infinity).
            # This tells Optuna's algorithm (TPE) that this area is "terrible" 
            # so it learns to avoid suggesting parameters in this region.
            return np.inf
        
        # If valid, add to our list of coordinates
        sensor_flat_list.extend([x, y, z])

    # --- C. Prepare Data for Physics Engine ---
    # The simulation requires a numpy matrix of shape (N, 3).
    # reshape(-1, 3) automatically handles the number of rows based on list length.
    sensor_coords = np.array(sensor_flat_list).reshape(-1, 3)
    
    # --- D. Run Simulation and Get Error ---
    try:
        # Call the actual physics simulation (from puff_simulation.py).
        # This returns the MAE (Mean Absolute Error).
        mae = run_real_puff_simulation(sensor_coords) 
        
    except Exception as e:
        # If the simulation crashes (e.g. numerical instability in C++),
        # we prune this trial so it doesn't stop the whole study.
        print(f"Simulation failed: {e}")
        raise optuna.TrialPruned()

    # Return the MAE. Optuna will try to find parameters that make this value smaller.
    return mae

# ==========================================
# Main Optimization Execution
# ==========================================
if __name__ == "__main__":

    # Define the study name (ID for the database)
    study_name = "sensor_optimization_v1"
    
    # Create (or load) the study.
    # direction="minimize" means we want the lowest possible MAE.
    # load_if_exists=True allows you to stop and restart the script without losing progress.
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  
        load_if_exists=True
    )

    print(f"Starting optimization...")

    # Run the optimization!
    # n_trials: How many different sensor configurations to try.
    # n_jobs: Number of parallel processes (8 CPUs). 
    #         Since our simulation is single-threaded, using n_jobs > 1 speeds up 
    #         optimization significantly by running multiple sims at once.
    study.optimize(optuna_objective, n_trials=500, n_jobs=8)

    print("\nOptimization Complete.")
    print(f"Best MAE found: {study.best_value}")
    print(f"Best Sensor Locations: {study.best_params}")