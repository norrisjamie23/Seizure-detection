"""
trainer.py

This module contains functions and utilities for training our model. 
It includes the training loop, validation loop, and early stopping functionalities.

The primary components are:
- train_one_epoch: Executes the training phase for one epoch.
- validate_one_epoch: Executes the validation phase for one epoch.
- train: The main function to coordinate training and validation.
- load: Function to load a model based on some weights.
"""

import torch
from torch import nn, optim
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Executes the training phase for one epoch.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader object for the training set.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimizer.
    - device (torch.device): The computing device (CPU or GPU).

    Returns:
    - float: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    train_loss_epoch = []  # List to store loss for each batch
    
    # Loop through each batch in the training DataLoader
    for batch in tqdm(train_loader, desc="Training", position=0, leave=True):

        # Move inputs to the selected computing device (CPU or GPU)
        inputs = batch.to(device)
                    
        # Zero out any pre-existing gradients to prevent accumulation
        optimizer.zero_grad()

        # Forward pass: Generate predictions from model
        outputs = model(inputs)
        
        # Calculate loss between predicted and actual values
        loss = criterion(inputs, outputs)

        # Backward pass: Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Append this batch's loss to the list for the epoch
        train_loss_epoch.append(loss.item())

    # Calculate and return the average training loss for this epoch
    return sum(train_loss_epoch) / len(train_loss_epoch)


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Executes the validation phase for one epoch.

    Parameters:
    - model (torch.nn.Module): The model to be validated.
    - val_loader (torch.utils.data.DataLoader): DataLoader object for the validation set.
    - criterion (torch.nn.Module): The loss function.
    - device (torch.device): The computing device (CPU or GPU).

    Returns:
    - float: The average validation loss for the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss_epoch = []  # List to store loss for each batch
    
    # Disable gradient calculation for efficiency in validation phase
    with torch.no_grad():
        # Loop through each batch in the validation DataLoader
        for batch in tqdm(val_loader, desc="Validation", position=0, leave=True):

            # Move inputs to the selected computing device (CPU or GPU)
            inputs = batch.to(device)
                        
            # Forward pass: Generate predictions from model
            outputs = model(inputs)
            
            # Calculate loss between predicted and actual values
            loss = criterion(inputs, outputs)
            
            # Append this batch's loss to the list for the epoch
            val_loss_epoch.append(loss.item())

    return sum(val_loss_epoch) / len(val_loss_epoch)


def train(train_loader, val_loader, model, device, weights_path):
    """
    The main function to coordinate training and validation.

    Parameters:
    - training_mode (bool): Flag to indicate whether to train the model.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - model (torch.nn.Module): The machine learning model.
    - device (torch.device): The computing device (CPU or GPU).
    - weights_path (str): The path to save/load model weights.

    """
    if training_mode:
        # Initialize loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training settings
        num_epochs = 100
        best_val_loss = float('inf')
        n_bad_epochs = 0
        
        # After this number of epochs without an improvement on val, training stops
        patience = 5

        # Loop through each epoch
        for epoch in range(num_epochs):
            # Execute one training and validation epoch; get average losses
            avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            avg_val_loss = validate_one_epoch(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            
            # Check and save the best model; implement early stopping
            if avg_val_loss < best_val_loss:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, weights_path)
                best_val_loss = avg_val_loss
                n_bad_epochs = 0  # Reset counter for bad epochs
            else:
                n_bad_epochs += 1  # Increment counter for bad epochs

            # Implement early stopping
            if n_bad_epochs >= patience:
                print("Early stopping triggered!")
                break

    # Load best model weights
    model = load(model, weights_path)

    return model


def load(model, weights_path):
    """
    Load the pre-trained model weights.

    Parameters:
    - model (torch.nn.Module): The PyTorch model for which the weights will be loaded.
    - weights_path (str): The path to the saved model weights.

    Returns:
    - torch.nn.Module: The model with loaded weights.
    """
    
    # Load model weights from the specified path
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
