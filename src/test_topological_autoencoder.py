import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from topological_autoencoder import TopologicalAutoencoder, SOMCriterion as som_criterion

# %%

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
    Trains the provided model using the specified parameters.

    Args:
        model: Neural network model to be trained.
        train_loader: Data loader containing the training dataset.
        training_block_start_epoch: Starting epoch for the training process.
        epochs_per_block: Number of epochs to train for.
        annealing_epochs: Number of epochs for annealing_factor.
        criterion: Loss function used for training.
        optimizer: Optimization algorithm.
        device: Device on which to run the training.

    Returns:
        None
    """
    model.to(device)  # Moves model to the specified device (e.g., GPU)
    
    for epoch in range(
        training_block_start_epoch,
        training_block_start_epoch + epochs_per_block,
    ):
        # Update standard deviation and learning rate
        sq2 = 0.05 * np.sqrt(2)
        std = 10 * (sq2 + (1 - sq2) * np.exp(-epoch / annealing_epochs))
        lr = np.exp(-epoch / annealing_epochs)
        
        model.topological_inner_layer.set_std(std)
        running_loss = 0.0
        
        # Iterate over the training data
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            _, z, outputs = model(images)
            
            # Calculate the loss using the provided criterion
            loss = criterion(images, outputs, z, lr)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print the training progress
        print(
            'Epoch {} - loss: {:12.8f} - std: {:5.3f} - lr: {:6.3f}'.format(
                epoch + 1,
                running_loss / len(train_loader),
                std,
                lr,
            )
        )

def test_model(model, test_loader, criterion, device):
    """
    Test a PyTorch model on a given test dataset.

    Args:
    model (nn.Module): The model to be tested.
    test_loader (DataLoader): The test data loader.
    criterion: The loss function to evaluate the model.
    device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
    None
    """
    model.to(device)
    
    # Test the model
    running_loss = 0.0
    model.eval()
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            _, z, outputs = model(images)
            
            loss = criterion(images, outputs, z, 0.0)
            running_loss += loss.item()
    
    print('Test loss: {}'.format(running_loss / len(test_loader)))

def visualize_results(model, test_loader, device):
    """
    Visualizes the results of image reconstruction using a given model.
    
    Args:
    model (torch.nn.Module): The model for image reconstruction
    test_loader (torch.utils.data.DataLoader): DataLoader with test images
    device (str): The device to run the model on (e.g., 'cuda' or 'cpu')
    
    Returns:
    matplotlib.figure.Figure: Figure showing original and reconstructed images
    """
    model.to(device)
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    
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
            )
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    fig.canvas.draw()

    return fig

def generate_from_inner(model):
    """
    Generates an image from the inner layer of the model and saves it as 'generated.png'.

    Args:
        model: The trained model used for generating the image.

    Returns:
        None
    """
    
    import matplotlib.pyplot as plt
    
    # Close any existing plots
    plt.close("all")
    
    # Get the indices of the topological inner layer
    idcs = model.topological_inner_layer.radial.grid
    
    # Compute the backward operation on the inner layer
    z = model.topological_inner_layer.backward(idcs.squeeze())
    
    # Apply ReLU activation
    z = model.relu(z.T)
    
    # Decode the inner layer to generate the image
    outs = model.decoder(z)
    
    # Convert the outputs to numpy array
    outs = outs.cpu().detach().numpy()
    
    # Reshape the outputs to form the final image
    outs = (
        outs.squeeze()
        .reshape(10, 10, 28, 28)
        .transpose(0, 2, 1, 3)
        .reshape(10 * 28, 10 * 28)
    )

    # Display the image
    fig, ax = plt.subplots(1, 1)
    ax.imshow(outs)
    
    # Save the generated image
    fig.savefig("generated.png")

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

    channels = [6, 16]
    kernel = 3
    initial_layer_side = 28
    inner_dim = 200
    topological_dim = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
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

    # Create the model
    model = TopologicalAutoencoder(
        initial_layer_side=initial_layer_side,
        inner_dim=inner_dim,
        topological_dim=topological_dim,
        channels=channels,
        kernel=kernel,
    ).to(device)

    # iEventually reload from file

    storage_path = 'stored_model'
    if os.path.isfile(storage_path):

        model.load(storage_path)
        model.eval()

        # Visualize the results
        fig = visualize_results(model, test_loader, device)
        fig.savefig(f'sim.png')

    else:

        # Define the loss function and optimizer
        criterion = som_criterion(beta=beta)
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
                total_epochs*annealing_factor,
                criterion,
                optimizer,
                device,
            )

            # Test the model
            test_model(model, test_loader, criterion, device)

            # Visualize the results
            fig = visualize_results(model, test_loader, device)
            fig.savefig(f'sim-{training_block_start_epoch:06d}.png')

            # test prototypes
            model.save(storage_path)

            generate_from_inner(model)

generate_from_inner(model)
