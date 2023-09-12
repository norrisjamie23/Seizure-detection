"""
datasets.py

This module contains utilities for converting EDF file paths and applying smoothing functions to 2D arrays.
The primary components are:
- convert_to_baseline_path: Converts an EDF file path for a seizure to its corresponding baseline file path.
- one_sided_moving_average_2D: Applies one-sided moving average smoothing to a 2D NumPy array.
"""

import numpy as np
import os
import json

# For plotting graphs
import matplotlib.pyplot as plt

def convert_to_baseline_path(original_path: str) -> str:
    """
    Convert an EDF file path for a seizure to its corresponding baseline file path.

    Parameters:
    - original_path (str): The original file path for the seizure EDF file.

    Returns:
    - str: The modified file path for the baseline EDF file.
    """

    # Split the original path into directory and file name
    directory, file_name = original_path.rsplit('/', 1)
    
    # Replace "Seizure" with "Sz" and append "_Baseline" to file name
    modified_file_name = file_name.replace("Seizure", "Sz")[:-4] + "_Baseline.EDF"
    
    # Combine directory and modified file name to create the new path
    new_path = f"{directory}/{modified_file_name}"

    return new_path


def one_sided_moving_average_2D(data, window_size):
    """
    Applies one-sided moving average smoothing to a 2D NumPy array.

    Parameters:
    - data (np.ndarray): The 2D NumPy array to be smoothed.
    - window_size (int): The size of the moving window to be applied.

    Returns:
    - np.ndarray: The smoothed 2D NumPy array.
    """
    
    # Initialize an empty array to store the smoothed data
    num_chans, time_steps = data.shape
    smoothed = np.zeros((num_chans, time_steps))
    
    # Loop through each channel and time step to apply smoothing
    for chan in range(num_chans):
        for t in range(time_steps):
            start_idx = max(0, t - window_size + 1)
            smoothed[chan, t] = np.mean(data[chan, start_idx:t+1])
            
    return smoothed