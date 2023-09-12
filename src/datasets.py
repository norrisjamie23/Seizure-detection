"""
datasets.py
-----------
This module provides functionalities for loading and preparing iEEG datasets.

Classes:
--------
- iEEGDataset

Functions:
----------
- split_data(directory_path: str, pattern: str, train_len: float = 0.8)
- load_and_prepare_data(seizure_file: str, standardise: bool = True)
- get_seizure_scores(seizure_file: str, model: torch.nn.Module, device: str, standardise: bool = True)
- plot_seizure_scores(seizure_file :str, model: torch.nn.Module, device: str, standardise: bool, window_size: int, save_path: str)
"""

from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split

import glob
import os

import mne
import numpy as np
import torch
import pandas as pd

# For progress bars
from tqdm import tqdm
from utils import *

class iEEGDataset(Dataset):
    def __init__(self, raw, fs, mean=None, std=None, notch_freq=60, epoch_dur=1, bandpass=[5, 50]):
        """
        A PyTorch Dataset class for loading and preprocessing SEEG data.

        Parameters:
        - raw : Raw instance from MNE
            The raw EEG data.
        - fs : int
            Desired sampling frequency.
        - mean : float
            Mean value for normalization.
        - std : float
            Standard deviation value for normalization.
        - notch_freq : int
            Frequency to apply notch filter at. Default is 60.
        - epoch_dur : int, optional
            Duration of each epoch/window. Default is 1.
        - bandpass : list of int, optional
            The frequency range for bandpass filter. Default is [5, 50].
        """

        # Filter out channels containing the string "Ref"
        picks = [chan for chan in raw.info['ch_names'] if "Ref" not in chan]

        # Retain only the channels in the "picks" list
        raw.pick(picks, verbose=False)

        # Apply notch and bandpass filters, and resample the data
        notch_filters = np.arange(notch_freq, bandpass[1], notch_freq)
        if len(notch_filters) > 0:
            raw.notch_filter(notch_filters, picks=picks, filter_length='auto', phase='zero', verbose=False)
        raw.filter(bandpass[0], bandpass[1], fir_design='firwin', verbose=False)
        raw.resample(fs, npad="auto")

        if mean is None and std is None:
            raw_data = raw.get_data()
            mean = np.mean(raw_data)
            std = np.std(raw_data, axis=1).mean()
        
        # Initialize instance variables
        self.raw = raw
        self.mean = mean
        self.std = std
        self.epoch_dur = epoch_dur
        self.windows = self._create_windows()

    def __len__(self):
        """
        Returns the total number of channel-window combinations.
        """
        return len(self.windows) * len(self.raw.info['ch_names'])

    def __getitem__(self, idx):
        """
        Return data for a given index.

        Parameters:
        - idx : int
            The index to retrieve data.

        Returns:
        - tensor : torch.Tensor
            The EEG data for the given index.
        """
        channel_idx = idx // len(self.windows)
        window_idx = idx % len(self.windows)
        win_start, win_end = self.windows[window_idx]
        X = self.raw.get_data(picks=channel_idx, start=win_start, stop=win_end)
        X = (X - self.mean) / self.std

        return torch.tensor(X, dtype=torch.float32)

    def _create_windows(self):
        """
        Create windows based on the specified epoch duration.
        
        Returns:
        - windows : list of tuple
            List of (start, end) sample tuples for each window.
        """
        win_len = int(self.epoch_dur * self.raw.info['sfreq'])
        starts = np.arange(0, len(self.raw.times) - win_len, win_len)
        ends = starts + win_len
        return list(zip(starts, ends))

def split_data(directory_path, pattern, train_len=0.8):
    """
    Splits the dataset into training and validation subsets.

    Parameters:
    -----------
    directory_path : str
        The directory containing the SEEG data files.
    pattern : str
        File pattern to search for.
    train_len : float, optional
        Proportion of data to use for training (default is 0.8).

    Returns:
    --------
    train_loader, val_loader : DataLoader, DataLoader
        Data loaders for training and validation sets.
    """

    # Fetch all file paths matching the given pattern
    file_paths = glob.glob(os.path.join(directory_path, pattern), recursive=True)

    all_train_datasets = []
    all_val_datasets = []

    for file_path in tqdm(file_paths):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        channels_file = file_path[:-8] + "channels.tsv"
        channels_df = pd.read_csv(channels_file, delimiter='\t')

        # Your code to compute the arrays
        mean_value = (np.array(list(channels_df.name)) == np.array([item.split()[-1] for item in raw.info['ch_names']])).mean()
        
        if mean_value < 0.8:
            print(f"Channels file {channels_file} doesn't appear to match edf file {file_path}, proceeding to next iteration. ({mean_value})")
            continue  # Go to the next iteration

        # Convert list to a NumPy array
        ch_names_array = np.array(raw.info['ch_names'])

        # Create the boolean mask from the Pandas DataFrame
        bool_mask = (channels_df.status == 'good').to_numpy()

        # Use boolean indexing on the NumPy array
        picks = ch_names_array[bool_mask]
        
        # Keep valid channels
        raw.pick(picks.tolist(), verbose=False)

        # Create a dataset for this patient
        dataset = iEEGDataset(raw, 512)
        
        # Split into train and val
        train_size = int(train_len * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)

    # Combine all training and validation datasets
    train_dataset_combined = ConcatDataset(all_train_datasets)
    val_dataset_combined = ConcatDataset(all_val_datasets)
    train_loader = DataLoader(train_dataset_combined, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset_combined, batch_size=32)

    return train_loader, val_loader


def load_and_prepare_data(seizure_file, standardise=True):
    """
    Loads and optionally standardizes SEEG data from an EDF file.

    Parameters:
    -----------
    seizure_file : str
        Path to the SEEG EDF file.
    standardise : bool, optional
        Whether to standardise the data based on interictal periods (default is True).

    Returns:
    --------
    DataLoader, list of channel labels.
    """
    
    # If data needs to be standardised, load the corresponding interictal file
    if standardise:
        # Convert seizure file path to corresponding interictal file path
        raw_interictal_file = convert_to_baseline_path(seizure_file)
        
        # Load interictal raw data
        raw_interictal = mne.io.read_raw_edf(raw_interictal_file, preload=True, verbose=False)
        
        # Create a dataset using the interictal data
        interictal_dataset = iEEGDataset(raw_interictal, 512, notch_freq=50)

    # Load raw seizure data
    raw = mne.io.read_raw_edf(seizure_file, preload=True, verbose=False)
    
    # Create a dataset using the seizure data, standardised if specified
    dataset = iEEGDataset(raw, 512, 
                          mean=interictal_dataset.mean if standardise else None,
                          std=interictal_dataset.std if standardise else None, 
                          notch_freq=50)

    # Extract channel labels excluding reference channels
    channel_labels = [chan for chan in raw.info['ch_names'] if "Ref" not in chan]
    
    # Create a DataLoader for the seizure dataset
    data_loader = DataLoader(dataset, batch_size=32)

    return data_loader, channel_labels


def get_seizure_scores(seizure_file, model, device, standardise=True):
    """
    Computes seizure scores for SEEG data based on a pretrained model.

    Parameters:
    -----------
    seizure_file : str
        Path to the SEEG EDF file.
    model : torch.nn.Module
        Pretrained PyTorch model.
    device : str
        Computational device ('cpu' or 'cuda').
    standardise : bool, optional
        Whether to standardise the data (default is True).

    Returns:
    --------
    np.ndarray, list
        Array of seizure scores, reshaped to (num_channels, num_samples), and list of channel labels.
    """

    # Get loader and channel labels
    data_loader, channel_labels = load_and_prepare_data(seizure_file, standardise)

    model.eval()
    mses = []
    with torch.no_grad():

        # Loop through seizure file
        for batch in data_loader:

            # Move inputs to the selected computing device (CPU or GPU)
            inputs = batch.to(device)
                        
            # Forward pass: Generate predictions from model
            outputs = model(inputs)
            
            # Calculate MSE
            mse_batch = np.mean(np.square(inputs.cpu().numpy() - outputs.cpu().numpy()), axis=-1)[:, 0]
            mses.extend(mse_batch)

    # Reshape to shape of seizure
    seizure_scores = np.array(mses).reshape(len(channel_labels), -1)

    return seizure_scores, channel_labels


def plot_seizure_scores(seizure_file, model, device, standardise=True, window_size=1, save_path=None):
    """
    Plot and optionally save the seizure scores for each channel and time point.
    
    Parameters:
    -----------
    seizure_file : str
        Path to the SEEG EDF file.
    model : torch.nn.Module
        Pretrained PyTorch model.
    device : str
        Computational device ('cpu' or 'cuda').
    standardise : bool, optional
        Whether to standardise the data (default is True).
    window_size : int, optional
        Size of the moving average window for smoothing the seizure scores (default is 1).
    save_path : str, optional
        Path to save the plot (default is None, meaning the plot won't be saved).

    Returns:
    --------
    None
    """
    
    # Compute seizure scores (Mean Squared Errors) using the get_seizure_scores function
    seizure_scores, channel_labels = get_seizure_scores(seizure_file, model, device, standardise)
    
    # Apply a one-sided moving average to smooth the seizure scores
    seizure_scores = one_sided_moving_average_2D(seizure_scores, window_size)
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    cax = ax.imshow(seizure_scores, aspect='auto')
    
    # Configure x-axis to represent time
    num_time_points = seizure_scores.shape[1]
    ax.set_xticks(np.arange(0, num_time_points, 5) - 0.5)
    ax.set_xticklabels(np.arange(0, num_time_points, 5))
    ax.set_xlabel("Time (s)")
    
    # Configure y-axis to represent SEEG channels
    ax.set_yticks(np.arange(len(channel_labels)))
    ax.set_yticklabels(channel_labels)
    ax.set_ylabel("SEEG Channel")
    
    # Add grid lines to the plot for better visibility of the individual cells
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # Set the title for the plot
    ax.set_title("Colour grid of seizure scores over time and channels")
    
    # Add colorbar to indicate the seizure score values
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('Seizure Score', rotation=270, labelpad=15)
    
    # Make sure all elements fit within the plot area
    plt.tight_layout()
    
    # Save the plot to disk if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)
        
    # Display the plot
    plt.show()
