import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from topological_maps import TopologicalMap, som_loss, stm_loss

class TopologicalAutoencoder(nn.Module):
    def __init__(
        self,
        inner_dim=2,
        topological_dim=100,
        channels=(6, 16),
        initial_layer_side=28,
        kernel=3,
    ):
        """Initializes a TopologicalAutoencoder model.

        Args:
        - inner_dim (int): Dimension of the inner hidden layer
        - topological_dim (int): Dimension of the topological layer
        - channels (tuple): Number of channels for each layer
        - initial_layer_side (int): Size of the initial layer side
        - kernel (int): Kernel size for the ConvAutoencoder
        """

        super(TopologicalAutoencoder, self).__init__()

        self.relu = nn.ReLU()

        channels = tuple(channels)

        # Calculate the layer sides and other dimensions
        layer_sides, toinner_dim, last_layer_channels, last_layer_side = calculate_layer_sides(
            channels, initial_layer_side, kernel=kernel)

        # Create a list of convolutional layers with specified channel numbers
        conv_layers = []
        for cur_ch, prev_ch in zip(channels, (1,) + channels[:-1]):
            conv_layers.extend([
                nn.Conv2d(prev_ch, cur_ch, kernel_size=kernel),
                nn.ReLU(inplace=True),
            ])

        # Define the encoder model using the list of convolutional layers, Flatten the output
        self.encoder = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
        )

        # TopologicalMap layer
        self.topological_inner_layer = TopologicalMap(inner_dim, topological_dim)

        self.linear_inner_layer = nn.Linear(toinner_dim, inner_dim)

        # Create a list of deconvolutional layers using the list of channel numbers
        deconv_layers = []
        for cur_ch, prev_ch in list(zip(channels, (1,) + channels[:-1]))[::-1]:
            deconv_layers.extend([
                nn.ConvTranspose2d(cur_ch, prev_ch, kernel_size=kernel),
                nn.ReLU(True),
            ])

        # Define the decoder network architecture using nn.Sequential
        self.decoder = nn.Sequential(
            nn.Linear(inner_dim, toinner_dim),
            nn.Unflatten(1, [last_layer_channels, last_layer_side, last_layer_side]),
            *deconv_layers,
        )

    def to(self, *args, **kwargs):
        """Move the model to the specified device."""
        self.topological_inner_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x):
        """Forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor

        Returns:
        - z (torch.Tensor): Inner hidden layer tensor
        - zt (torch.Tensor): Topological layer tensor
        - y (torch.Tensor): Decoded output tensor
        """
        x = self.encoder(x)
        z = self.linear_inner_layer(x)
        z = self.relu(z) 
        zt = self.topological_inner_layer(z)
        y = self.decoder(z)
        return z, zt, y

    def load(self, path):
        """Load model state from a saved file."""
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """Save the model state to a file."""
        torch.save(self.state_dict(), path)


def calculate_layer_sides(channels, initial_layer_side, kernel=3):
    """Calculate the layer sides and dimensions.

    Args:
    - channels (tuple): Number of channels for each layer
    - initial_layer_side (int): Size of the initial layer side
    - kernel (int): Kernel size for the ConvAutoencoder

    Returns:
    - layer_sides (list): List of layer side sizes
    - toinner_dim (int): Total inner dimension
    - last_layer_channels (int): Last layer channels
    - last_layer_side (int): Size of the last layer side
    """
    layer_sides = [initial_layer_side]
    for channel in channels:
        new_side = layer_sides[-1] - kernel + 1
        layer_sides.append(new_side)

    last_layer_side = layer_sides[-1]
    last_layer_channels = channels[-1]
    toinner_dim = (last_layer_side**2) * last_layer_channels

    return layer_sides, toinner_dim, last_layer_channels, last_layer_side


class SOMCriterion:
    """
    Criterion class for Self-Organizing Maps.
    The criterion combines Mean Squared Error Loss (MSE) and SOM loss using a beta parameter for weighting.

    Args:
    - beta (float): Weighting factor for combining MSE and SOM losses (default=1)

    Methods:
    - __init__: Initializes the SOMCriterion class with the specified beta value.
    - __call__: Calculates the combined loss for inputs, outputs, inners, and learning rate (lr).
    """
    def __init__(self, beta=1):
        """
        Initializes the SOMCriterion with the provided beta value.
        
        Args:
        - beta (float): Weighting factor for combining MSE and SOM losses (default=1)
        """
        self.mse_loss = nn.MSELoss()
        self.beta = beta

    def __call__(self, inputs, outputs, inners, lr):
        """
        Computes the combined loss based on MSE and SOM losses for the given inputs, outputs, inners, and learning rate.
        
        Args:
        - inputs: Input data for the criterion
        - outputs: Output data predicted by the model
        - inners: Inners data for SOM loss calculation
        - lr (float): Learning rate
        
        Returns:
        - Combined loss value
        """
        loss_mse = (1 - self.beta) * self.mse_loss(inputs, outputs)
        loss_som = self.beta * som_loss(inners)  # Need to define som_loss function
        return lr * (loss_mse + loss_som)


class STMCriterion:
    """
    Criterion class for Spatio-Temporal Memory Consolidation.
    The criterion combines Mean Squared Error Loss (MSE) and STM loss using a beta parameter for weighting.

    Args:
    - beta (float): Weighting factor for combining MSE and STM losses (default=1)
    - output_size (int): Size of the output (default=100)

    Methods:
    - __init__: Initializes the STMCriterion class with the specified beta and output_size values.
    - __call__: Calculates the combined loss for inputs, outputs, inners, targets, and learning rate (lr).
    """
    def __init__(self, beta=1, output_size=100):
        """
        Initializes the STMCriterion with the provided beta and output_size values.
        
        Args:
        - beta (float): Weighting factor for combining MSE and STM losses (default=1)
        - output_size (int): Size of the output data (default=100)
        """
        self.mse_loss = nn.MSELoss()
        self.beta = beta

    def __call__(self, inputs, outputs, inners, targets, lr):
        """
        Computes the combined loss based on MSE and STM losses for the given inputs, outputs, inners, targets, and learning rate.
        
        Args:
        - inputs: Input data for the criterion
        - outputs: Output data predicted by the model
        - inners: Inners data for STM loss calculation
        - targets: Target data for STM loss calculation
        - lr (float): Learning rate
        
        Returns:
        - Combined loss value
        """
        loss_mse = (1 - self.beta) * self.mse_loss(inputs, outputs)
        loss_stm = self.beta * stm_loss(inners, targets)  # Need to define stm_loss function
        return lr * (loss_mse + loss_stm)
