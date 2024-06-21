import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the convolutional autoencoder class
class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder class.

    Args:
        inner_dim (int): The inner dimension for autoencoder (default: 50).
        channels (tuple): Tuple representing the number of channels in each layer (default: (6, 16)).
        initial_layer_side (int): Initial layer side dimension (default: 28).
        kernel (int): Kernel size for convolutional layers (default: 3).
        inner_layer (nn.Module): Inner layer for the autoencoder (default: None).
    """

    def __init__(
        self,
        inner_dim=50,
        channels=(6, 16),
        initial_layer_side=28,
        kernel=3,
        inner_layer=None,
    ):
        super(ConvAutoencoder, self).__init__()
        channels = tuple(channels)

        (
            layer_sides,
            toinner_dim,
            last_layer_channels,
            last_layer_side,
        ) = calculate_layer_sides(channels, initial_layer_side, kernel=kernel)

        # Create a list of convolutional layers with specified channel numbers
        conv_layers = []
        for cur_ch, prev_ch in zip(channels, (1,) + channels[:-1]):
            conv_layers.extend(
                [
                    nn.Conv2d(prev_ch, cur_ch, kernel_size=kernel),
                    nn.ReLU(inplace=True),
                ]
            )

        # Define the encoder model using the list of convolutional layers, Flattening the output
        self.encoder = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
        )

        # Create a default inner layer if not specified
        self.inner = inner_layer
        if self.inner is None:
            self.inner = nn.Sequential(
                nn.Linear(toinner_dim, inner_dim), nn.ReLU(True)
            )

        # Create a list of deconvolutional layers using the list of channel numbers
        deconv_layers = []
        for cur_ch, prev_ch in list(zip(channels, (1,) + channels[:-1]))[::-1]:
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(cur_ch, prev_ch, kernel_size=kernel),
                    nn.ReLU(True),
                ]
            )

        # Define the decoder network architecture using nn.Sequential
        self.decoder = nn.Sequential(
            nn.Linear(inner_dim, toinner_dim),
            nn.Unflatten(
                1, [last_layer_channels, last_layer_side, last_layer_side]
            ),
            *deconv_layers,
        )


    def forward(self, x):
        """Forward pass function of the autoencoder.

        Args:
            x (Tensor): Input tensor to the autoencoder.

        Returns:
            Tensor: Encoded tensor representation.
            Tensor: Reconstructed output tensor.
        """
        x = self.encoder(x)
        z = self.inner(x)
        y = self.decoder(z)
        return z, y


def calculate_layer_sides(channels, initial_layer_side, kernel=3):
    """
    Calculate the side length of each layer and the total inner dimension based on the input channels and parameters.

    Args:
        channels (list): List of channel sizes in each layer
        initial_layer_side (int): Initial side length of the input layer
        kernel (int, optional): Size of the kernel used in the convolution operation. Default is 3.
    
    Returns:
        tuple: A tuple containing the calculated layer sides, total inner dimension, last layer channels, 
               and last layer side length.
    """
    
    # Calculate the layer side for each channel
    layer_sides = [initial_layer_side]
    for channel in channels:
        new_side = layer_sides[-1] - kernel + 1
        layer_sides.append(new_side)

    # Calculate the total inner dimension based on the final side length and last channel size
    last_layer_side = layer_sides[-1]
    last_layer_channels = channels[-1]
    toinner_dim = (last_layer_side**2) * last_layer_channels

    return layer_sides, toinner_dim, last_layer_channels, last_layer_side


