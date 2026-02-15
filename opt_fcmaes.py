import numpy as np
import os, sys
# Legacy print options for cleaner numpy output in logs
np.set_printoptions(legacy='1.25') 

# Import the simulation factory and validation logic
from puff_simulation import get_objective, is_valid

# Scipy bounds are used to define the search space limits
from scipy.optimize import Bounds

# Import specific components from fcmaes:
# - wrapper: Helps align our objective function signature with what fcmaes expects.
# - Bite_cpp: The specific optimization algorithm (Bi-Population Evolution Strategy).
#             It is excellent for multimodal problems (many local minima).
# - retry: A meta-algorithm that restarts the optimizer multiple times in parallel 
#          to find the global minimum and avoid getting stuck.
from fcmaes.optimizer import wrapper, Bite_cpp, de
from fcmaes import retry

# Loguru is used for high-performance, thread-safe logging, which is essential
# when running parallel optimizations that print to stdout.
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}", level="INFO")

def evolutionary_optimizer(fitness, lb, ub, x0=None,
                           num_retries=8, 
                           max_evals=1000,
                           workers=8):
    """
    Configures and runs the parallel evolutionary optimization.
    
    Args:
        fitness (callable): The objective function to minimize.
        lb (list): Lower bounds for decision variables.
        ub (list): Upper bounds for decision variables.
        x0 (list, optional): Initial guess for the parameters.
        num_retries (int): Number of parallel optimization 'threads' to run.
                           fcmaes runs multiple independent optimizations simultaneously.
        max_evals (int): Maximum allowed function evaluations (budget).
        workers (int): Number of CPU cores to utilize.
    """
    
    # We use the 'retry.minimize' function, which is the powerhouse of fcmaes.
    # It doesn't just run once; it runs 'num_retries' times in parallel.
    # If one run gets stuck in a local minimum, others might succeed.
    res = retry.minimize(
        # The wrapper logs progress and handles exception safety
        wrapper(fitness), 
        
        # Define the search space constraints
        bounds=Bounds(lb, ub),
        
        # How many independent optimization runs to execute
        num_retries=num_retries,
        
        # The specific algorithm to use. 
        # Bite_cpp is a C++ implementation of a method similar to Differential Evolution
        # but optimized for lower-dimensional, expensive objective functions.
        # It handles bound constraints natively.
        optimizer=Bite_cpp(max_evals, guess=x0),
        
        # Parallelization control
        workers=workers
    )
    return res

# Initialize the physics engine once (outside the loop)
run_real_puff_simulation = get_objective()

# ==========================================
# Objective Function Wrapper
# ==========================================

def fcmaes_objective(x):
    """
    The main callback for the optimizer.
    
    Unlike Optuna (which passes a 'trial' object), fcmaes passes a simple 
    numpy array 'x' containing the current candidate solution (decision variables).
    """
    
    # 1. Decode the decision variables
    #    'x' is a flat array of floats provided by the optimizer.
    sensor_flat_list = x
    
    #    Reshape into (N, 3) matrix for the physics engine.
    #    -1 lets numpy figure out the number of sensors based on input length.
    sensor_coordinates = np.array(sensor_flat_list).reshape(-1, 3)

    # 2. Apply Hard Constraints (Position Filtering)
    #    We iterate through every proposed sensor location.
    #    If ANY sensor is in an invalid location (inside an obstacle/cube),
    #    we immediately return infinity.
    #    This saves time by avoiding the expensive physics simulation for invalid layouts.
    for sx, sy, sz in sensor_coordinates:
        if not is_valid(sx, sy, sz): 
            # Return infinity to tell the optimizer this is a forbidden region.
            return np.inf

    # 3. Run Physics Simulation
    try:
        # Pass the valid coordinates to the Gaussian Puff model.
        # This returns the Mean Absolute Error (MAE).
        mae = run_real_puff_simulation(sensor_coordinates) 
        
    except Exception as e:
        # Robustness: If the C++ simulation backend crashes (e.g. math error),
        # log it and return infinity so the optimizer moves away from this parameter set.
        print(f"Simulation failed: {e}")
        return np.inf

    return mae

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Define Search Space
    #    We are optimizing 2 sensors, each has (x, y, z). Total = 6 variables.
    #    Bounds are [-120, 120] for every dimension.
    #    We multiply the list by 6 to create a list of length 6.
    lb = [-120.0] * 6  # Lower bounds: [-120, -120, -120, -120, -120, -120]
    ub = [120.0] * 6   # Upper bounds: [ 120,  120,  120,  120,  120,  120]
    
    print(f"Starting optimization with fcmaes...")
    
    # 2. Run the Evolutionary Optimizer
    #    This blocks until all retries are finished or the evaluation budget is met.
    res = evolutionary_optimizer(
        fcmaes_objective, 
        lb, 
        ub,
        num_retries=16,   # Run 16 independent searches
        workers=8,        # Use 8 threads
        max_evals=2000    # Total budget of 2000 simulations
    )

    print("\nOptimization Complete.")
    print(f"Best MAE found: {res.fun}")
    
    # 3. Display Best Coordinates
    #    Reshape the flat result array back to (N, 3) for readability
    best_coords = np.array(res.x).reshape(-1, 3)
    print(f"Best Sensor Locations:\n{best_coords}")