import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from topological_autoencoder import TopologicalAutoencoder, STMCriterion as stm_criterion

def train_model(
    model,
    train_loader,
    training_block_start_epoch,
    epochs_per_block,
    annealing_epochs,
    criterion,
    optimizer,
    device,
):
    """
    Training function for a given model using the specified parameters.

    Args:
        model: Neural network model to be trained.
        train_loader: DataLoader for training dataset.
        training_block_start_epoch: Starting epoch for training block.
        epochs_per_block: Number of epochs for training block.
        annealing_epochs: Number of epochs for annealing_factor.
        criterion: Loss function criterion.
        optimizer: Optimizer for model parameters.
        device: The device on which the model is trained (e.g., 'cuda' for GPU).

    Returns:
        None
    """
    
    model.to(device)  # Move model to specified device
    
    for epoch in range(training_block_start_epoch, training_block_start_epoch + epochs_per_block):
        # annealing_factor with standard deviation and learning rate
        sq2 = 0.05 * np.sqrt(2)
        std = 10 * (sq2 + (1 - sq2) * np.exp(-epoch / annealing_epochs))
        lr = np.exp(-epoch / annealing_epochs)

        model.topological_inner_layer.set_std(std)
        running_loss = 0.0
        
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            optimizer.zero_grad()
            _, z, outputs = model(images)
            zt = model.topological_inner_layer.radial(points[labels], std, as_point=True)
            
            # Compute loss and backpropagate
            loss = criterion(images, outputs, z, zt, lr)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print epoch statistics
        print('Epoch {} - loss: {:12.8f} - std: {:5.3f} - lr: {:6.3f}'.format(
            epoch + 1,
            running_loss / len(train_loader),
            std,
            lr
        ))

def test_model(model, test_loader, criterion, device):
    """
    Test the given model using the test_loader.

    Args:
    - model: The model to be tested.
    - test_loader: Data loader for the test data.
    - criterion: The loss criterion function.
    - device: Device to run the model on (CPU or GPU).

    Returns:
    None
    """
    # Move model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Get the standard deviation from the model's topological inner layer
    std = model.topological_inner_layer.curr_std

    # Disable gradient tracking during inference
    with torch.no_grad():
        running_loss = 0.0
        for data in test_loader:
            images, labels = data
            images = images.to(device)  # Move input data to the specified device
            _, z, outputs = model(images)

            # Compute the radial transformation
            zt = model.topological_inner_layer.radial(points[labels], std, as_point=True)

            # Calculate the loss using the given criterion
            loss = criterion(images, outputs, z, zt, 0.0)
            running_loss += loss.item()

    # Print the test loss
    print('Test loss: {}'.format(running_loss / len(test_loader)))

def visualize_results(model, test_loader, device):
    """Visualize original images and reconstructed images
    
    Args:
        model: A trained model for image reconstruction
        test_loader: DataLoader containing test images
        device: Device to use for processing (e.g., 'cuda' for GPU)
        
    Returns:
        fig: Matplotlib figure showing original and reconstructed images
    """
    model.to(device)  # Move model to the specified device

    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)  # Move data to the specified device

    # Generate reconstructed images
    *_, outputs = model(images)

    plt.close("all")
    # Plot the original images and reconstructed images
    fig, axes = plt.subplots(
        nrows=2, ncols=10, sharex=True, sharey=True, figsize=(8, 2)
    )

    for images, row in zip([images, outputs], axes):
        for img, ax in zip(images, row):
            ax.imshow(
                img.cpu().detach().numpy().squeeze(), cmap='gray'
            )  # Move the image back to CPU for plotting
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.canvas.draw()

    return fig

def generate_from_inner(model):
    """Generate synthesized image from the inner layer
    
    Args:
        model: A trained model containing the inner layers
    
    Notes:
        This function generates an image from the inner layers of the model.
        The synthesized image is saved as 'generated-stm.png'.
    """
    plt.close("all")
    idcs = model.topological_inner_layer.radial.grid
    z = model.topological_inner_layer.backward(idcs.squeeze())
    z = model.relu(z.T)
    outs = model.decoder(z)
    outs = outs.cpu().detach().numpy()
    outs = (
        outs.squeeze()
        .reshape(10, 10, 28, 28)
        .transpose(0, 2, 1, 3)
        .reshape(10 * 28, 10 * 28)
    )

    fig, ax = plt.subplots(1, 1)
    ax.imshow(outs)
    fig.savefig("generated-stm.png")

# %%

if __name__ == '__main__':

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('agg')

    # Define hyperparameters
    # Total number of epochs for training
    total_epochs = 20
    # annealing_factor factor for adjusting learning rate
    annealing_factor = 0.2
    # Number of epochs in each training block before evaluation
    epochs_per_block = 2
    # Batch size used for data loading during training
    batch_size = 64
    # Learning rate used by the optimizer
    learning_rate = 0.001
    # Beta parameter for optimization
    beta = 0.2

    # Network architecture parameters
    channels = [6, 16]
    kernel = 3
    initial_layer_side = 28
    inner_dim = 200
    topological_dim = 100

    # Determine device - GPU or CPU based on availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Defining target points for topological labeling
    points = torch.Tensor(
        [
            [0.15, 0.17],
            [0.12, 0.54],
            [0.16, 0.84],
            [0.50, 0.15],
            [0.36, 0.45],
            [0.62, 0.50],
            [0.48, 0.82],
            [0.83, 0.17],
            [0.88, 0.50],
            [0.83, 0.83],
        ]
    ).to(device) * 10

    # Load the MNIST dataset
    train_dataset = MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_dataset = MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create an instance of TopologicalAutoencoder model
    model = TopologicalAutoencoder(
        initial_layer_side=initial_layer_side,
        inner_dim=inner_dim,
        topological_dim=topological_dim,
        channels=channels,
        kernel=kernel,
    ).to(device)

    # Define the path to stored model
    storage_path = 'stored_model_stm'

    if os.path.isfile(storage_path):
        # Load the stored model
        model.load(storage_path)
        model.eval()

        # Visualize the results
        fig = visualize_results(model, test_loader, device)
        fig.savefig(f'sim.png')
    else:
        # Define the loss function and optimizer
        criterion = stm_criterion(beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for training_block_start_epoch in range(
            0, total_epochs, epochs_per_block
        ):
            # Train the model
            train_model(
                model,
                train_loader,
                training_block_start_epoch,
                epochs_per_block,
                total_epochs * annealing_factor,
                criterion,
                optimizer,
                device,
            )

            # Test the model
            test_model(model, test_loader, criterion, device)

            # Visualize the results
            fig = visualize_results(model, test_loader, device)
            fig.savefig(f'sim-stm-{training_block_start_epoch:06d}.png')

            # Save the current model
            model.save(storage_path)

            # Generate outputs from the inner layer
            generate_from_inner(model)

    # Generate outputs from the inner layer
    generate_from_inner(model)
