"""
anomaly_detector.py

This module contains the AnomalyDetector class, a  convolutional autoencoder
specifically designed for seizure detection in iEEG data.

The primary components are:
- __init__: Initializes the model architecture.
- forward: Defines the forward pass.
"""

import torch.nn as nn
import torch.nn.functional as F

class AnomalyDetector(nn.Module):
    """
    An improved convolutional autoencoder model for seizure detection in EEG data.
    """
    def __init__(self, input_channels):
        """
        Initialize the AnomalyDetector model.
        Parameters:
        - input_channels (int): Number of input channels.
        """
        super(AnomalyDetector, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)

        # Batch Normalization layers for encoder
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Decoder layers
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        # Batch Normalization layers for decoder
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        """
        Forward pass through the model.
        Parameters:
        - x (torch.Tensor): Input tensor.
        Returns:
        - x (torch.Tensor): Reconstructed output tensor.
        """

        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Decoder
        x = F.relu(self.bn4(self.deconv1(x)))
        x = F.relu(self.bn5(self.deconv2(x)))
        x = self.deconv3(x)
        
        return x
