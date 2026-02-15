from FastGaussianPuff import GaussianPuff
import numpy as np
import pandas as pd
import random


#See https://gemini.google.com/share/504ca40d3887

# Define the maximum dimension for the validation grid. 
# This defines a cubic domain size (e.g., 240 meters per side if mapped 1:1).
GRID_MAX = 240

def compute_valid_position_array(seed=42):
    """
    Creates a 3D boolean mask representing valid and invalid regions in space.
    This is used to constrain the optimization to specific areas (or avoid obstacles).
    """
    # 1. Initialize the boolean array with 'False'.
    #    Dimensions are GRID_MAX + 1 to include the boundary index.
    grid_dim = GRID_MAX + 1
    valid_positions_array = np.zeros((grid_dim, grid_dim, grid_dim), dtype=bool)

    # 2. Define parameters for generating random geometric shapes ("cubes").
    num_cubes = 50       # Number of shapes to generate
    min_size = 50        # Minimum side length of a cube
    max_size = 100       # Maximum side length of a cube

    # 3. Fill the array with these cubes.
    #    We use a fixed seed to ensure the "map" is the same every time we run the script.
    random.seed(seed) 

    for _ in range(num_cubes):
        # Generate random dimensions and position for the cube
        size = random.randint(min_size, max_size)
        
        # Calculate random starting coordinates.
        # We ensure the cube fits within the grid boundaries (grid_dim - size).
        x_start = random.randint(0, grid_dim - size)
        y_start = random.randint(0, grid_dim - size)
        z_start = random.randint(0, grid_dim - size)
        
        # Set the voxels within this cube to 'True'.
        # In the 'is_valid' logic below, these True regions will be treated 
        # as either valid zones or obstacles depending on the boolean logic used there.
        valid_positions_array[
            x_start : x_start + size, 
            y_start : y_start + size, 
            z_start : z_start + size
        ] = True
        
    return valid_positions_array

# Pre-compute the validation array when the module is loaded.
valid_positions_array = compute_valid_position_array()

def is_valid(x, y, z):
    """
    Checks if a specific 3D coordinate (x, y, z) is a valid location for a sensor.
    This function maps continuous coordinates (floats) to the discrete boolean grid.
    
    Args:
        x, y, z: Float coordinates from the optimizer (range approx -120 to 120).
        
    Returns:
        bool: True if the position is valid, False otherwise.
    """
    # 1. Offset logic:
    #    The optimizer explores a centered space (e.g., -120 to +120).
    #    The numpy array uses positive indices (0 to 240).
    #    We add an offset (120) to shift the coordinate origin to the array center.
    offset = GRID_MAX / 2
    ix, iy, iz = int(x + offset), int(y + offset), int(z + offset)

    # 2. Boundary Check:
    #    Ensure the calculated indices actually fall inside the numpy array.
    if (0 <= ix <= GRID_MAX) and (0 <= iy <= GRID_MAX) and (0 <= iz <= GRID_MAX):
        # 3. Boolean Logic:
        #    'valid_positions_array' has True inside the cubes.
        #    Here we return 'not ...', which means:
        #    - If inside a cube (True) -> Return False (Invalid/Obstacle)
        #    - If outside a cube (False) -> Return True (Valid/Free Space)
        return not valid_positions_array[ix, iy, iz]

    # If the coordinate is outside the defined grid bounds, default to True (Valid).
    # (Depending on requirements, you might want to return False here to enforce bounds).
    return True

def objective_function_mae(flat_sensor_coords, *args):
    """
    The Core Physics Objective Function.
    Calculates the Mean Absolute Error (MAE) between the model predictions
    at the proposed sensor locations and the 'ground truth'.
    
    Args:
        flat_sensor_coords (list/array): A flat list of coordinates [x1, y1, z1, x2, ...]
                                         provided by the optimizer.
        args: A tuple containing the simulation environment (wind, source, ground truth).
    """
    
    # 1. Unpack the fixed simulation parameters provided by 'get_objective'.
    (true_concentrations, wind_s, wind_d, source_coords, 
     emission_rate, t_start, t_end, dt_params) = args
     
    # 2. Reshape Decision Variables:
    #    Optuna works with individual float parameters. We flatten them into a list.
    #    Here, we reshape that flat list back into an (N_sensors, 3) matrix
    #    that the GaussianPuff API expects.
    #    '-1' infers the number of sensors automatically based on input length.
    sensor_coordinates = np.array(flat_sensor_coords).reshape(-1, 3)
    
    # 3. Configure the Gaussian Puff Model:
    #    This initializes the C++ backend with the environment data.
    gp_model = GaussianPuff(
        obs_dt=dt_params['obs_dt'],       # Resolution of wind data (e.g., 60s)
        sim_dt=dt_params['sim_dt'],       # Internal physics timestep (e.g., 1s)
        puff_dt=dt_params['puff_dt'],     # Frequency of puff release (e.g., 1s)
        simulation_start=t_start,         # Start datetime (UTC)
        simulation_end=t_end,             # End datetime (UTC)
        time_zone="UTC",                  
        source_coordinates=source_coords, # Location of the gas leak
        emission_rates=[emission_rate],   # Rate of the leak
        wind_speeds=wind_s,               # Wind speed time series
        wind_directions=wind_d,           # Wind direction time series
        
        # EFFICIENCY FLAG: 
        # 'using_sensors=True' is critical. It tells the model to ONLY compute 
        # concentrations at the specific [x,y,z] points in 'sensor_coordinates'.
        # Without this, it would compute a dense 3D grid, which is 1000x slower.
        using_sensors=True,               
        sensor_coordinates=sensor_coordinates, 
        
        quiet=True                        # Suppress verbose C++ logging
    )

    # 4. Run the Simulation:
    #    gp_model.simulate() executes the puff advection-diffusion loop.
    try:
        predicted_concentrations = gp_model.simulate()
    except Exception as ex:
        # Robustness: If the physics engine crashes (e.g., NaN coordinates),
        # return a massive error score to tell the optimizer this solution is bad.
        return np.inf

    # 5. Compute Error (MAE):
    #    Compare the 'predicted_concentrations' (what sensors at these new locations see)
    #    vs 'true_concentrations' (what we *expect* to see based on ground truth).
    #    Note: In a real placement problem, 'true_concentrations' would be derived 
    #    from a high-fidelity reference model (like LES) or historical data.
    
    if predicted_concentrations.shape != true_concentrations.shape:
        # Fallback if shapes mismatch (e.g. slight timing diffs), flatten both to compare.
        predicted_flat = predicted_concentrations.flatten()
        true_flat = true_concentrations.flatten()
    else:
        predicted_flat = predicted_concentrations
        true_flat = true_concentrations

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(predicted_flat - true_flat))
    
    return mae

def get_objective():
    """
    Factory function to setup the 'Ground Truth' environment and return
    a callable objective function ready for the optimizer.
    """
    
    # --- 1. Define Environmental Constraints ---
    obs_dt = 60.0  # Wind updates every 60 seconds
    duration_min = 10
    # Calculate number of data points needed
    n_steps = int(duration_min * 60 / obs_dt) + 1
    
    # --- 2. Create Synthetic Wind Data ---
    # Generate a sine wave for wind speed (varying between ~2 and 7 m/s)
    wind_speeds = np.abs(np.sin(np.linspace(0, 3, n_steps)) * 5) + 2  
    # Constant wind direction (270 degrees = blowing from West)
    wind_directions = np.full(n_steps, 270.0)  
    
    # Define Time Range (UTC is required by the library)
    start = pd.Timestamp("2022-01-01 12:00", tz="UTC")
    end = start + pd.Timedelta(minutes=duration_min)
    
    params = {
        'obs_dt': 60.0,
        'sim_dt': 1.0,   # High-res physics step
        'puff_dt': 1.0   # Release a puff every second
    }
    
    # Define Source: Location [0,0,2] meters, Rate 5.0 kg/hr
    source_loc = np.array([[0.0, 0.0, 2.0]]) 
    emission_q = 5.0 
    
    # --- 3. Establish "Ground Truth" ---
    # We define where the "Perfect" sensors are located. 
    # The optimization goal is effectively to find locations that mimic the data 
    # collected by these sensors (or maximize detection probability in other formulations).
    true_sensors = np.array([[50.0, 0.0, 1.5], [100.0, 10.0, 1.5]])
    
    # Run the model ONCE with these "Truth" sensors to generate the reference data.
    truth_model = GaussianPuff(
        obs_dt=params['obs_dt'], sim_dt=params['sim_dt'], puff_dt=params['puff_dt'],
        simulation_start=start, simulation_end=end, time_zone="UTC",
        source_coordinates=source_loc, emission_rates=[emission_q],
        wind_speeds=wind_speeds, wind_directions=wind_directions,
        using_sensors=True, sensor_coordinates=true_sensors, quiet=True
    )
    ground_truth_data = truth_model.simulate()

    # --- 4. Package Arguments ---
    # Create a tuple of all fixed data needed for the objective function
    args = (ground_truth_data, wind_speeds, wind_directions, source_loc, 
            emission_q, start, end, params)   
    
    # Define the wrapper function that the optimizer will call.
    # It takes ONLY the variable coordinates, and passes the fixed 'args' internally.
    def run_gaussian_puff_simulation(flat_sensor_coords):
        error = objective_function_mae(flat_sensor_coords, *args)
        return error
    
    return run_gaussian_puff_simulation


if __name__ == "__main__":
    # Test block to verify the simulation works without Optuna
    run_gaussian_puff_simulation = get_objective()    
    
    # Define a random guess for sensor locations
    initial_sensor_guess = np.array([
        20.0, 20.0, 1.5,   # Sensor 1
        30.0, -20.0, 1.5   # Sensor 2
    ])

    # Calculate error for this guess
    error = run_gaussian_puff_simulation(initial_sensor_guess)
    
    print(f"Calculated MAE for initial guess: {error:.4f} ppm")