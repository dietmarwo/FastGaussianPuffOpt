import numpy as np
import os, sys
np.set_printoptions(legacy='1.25') 
from puff_simulation import get_objective, is_valid
from scipy.optimize import Bounds
from fcmaes.optimizer import wrapper, Bite_cpp, de
from journal import journal_wrapper
from fcmaes import retry
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}", level="INFO")

def evolutionary_optimizer(fitness, lb, ub, x0 = None,
        num_retries=8, 
        max_evals=1000,
        workers=8):
    
    # res = de.minimize(wrapper(fitness),
    #     bounds=Bounds(lb, ub), workers=8, popsize=16)
    bounds = Bounds(lb, ub)
    wrapped = journal_wrapper(wrapper(fitness), bounds, 
                        "fcmaes_journal.log", "Puff Simulation", study_id=0, batch_size=8)
    
    res = retry.minimize(
        wrapped, # for debugging optimization progress
        bounds=bounds,
        num_retries=num_retries,
        optimizer=Bite_cpp(max_evals, guess=x0),
        #optimizer=Cma_cpp(max_evals),
        workers=workers
    )
    return res

run_real_puff_simulation = get_objective()

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

def fcmaes_objective(x):
    """
    The fcmaes optimization target.
    """
    sensor_flat_list = x
    # Reshape for the physics engine
    sensor_coordinates = np.array(sensor_flat_list).reshape(-1, 3)
    for x, y, z in sensor_coordinates:
        if not is_valid(x, y, z): # invalid position?
            return np.inf

    try:
        # Call the actual objective function defined in previous turns
        mae = run_real_puff_simulation(sensor_coordinates) 
        # Call a mock objective
        # mae = run_mock_puff_simulation(sensor_coords)
    except Exception as e:
        # If simulation fails (e.g., numerical instability), prune the trial
        print(f"Simulation failed: {e}")
        return np.inf

    return mae

# ==========================================
# 3. Main Optimization
# ==========================================
if __name__ == "__main__":
    
    lb = [-120.0]*6 # lower bound coordinates
    ub = [120.0]*6 # upper bound coordinates
    
    res = evolutionary_optimizer(fcmaes_objective, lb, ub)

    print("\nOptimization Complete.")
    print(f"Best MAE: {res.fun}")
    print(f"Best Params: {res.x}")
        