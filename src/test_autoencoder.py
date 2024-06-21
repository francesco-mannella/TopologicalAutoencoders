import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from autoencoder import ConvAutoencoder


def train_model(model, train_loader, num_epochs, criterion, optimizer, device):
    """
    Trains the given model using the provided data loader, criterion, optimizer, and device.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader for training data.
        num_epochs (int): The number of epochs to train the model.
        criterion (loss function): The loss function used for optimization.
        optimizer (Optimizer): The optimizer for updating model parameters.
        device (str): The device to perform training on (e.g., 'cuda' for GPU).

    Returns:
        None
    """
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            'Epoch {} - Training loss: {}'.format(
                epoch + 1, running_loss / len(train_loader)
            )
        )


def test_model(model, test_loader, criterion, device):
    """
    Evaluate the given model using the provided data loader and criterion on
    the specified device.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        criterion: The loss function used for evaluation.
        device (str): The device on which to perform the evaluation.

    Returns:
        None
    """
    model.to(device)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            _, outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item()
    print('Test loss: {}'.format(test_loss / len(test_loader)))


def visualize_results(model, test_loader, device):
    """
    Visualizes the results of a model by plotting original and reconstructed images.

    Args:
    model (nn.Module): The trained model
    test_loader (DataLoader): DataLoader containing test images
    device (str): Device to use ('cpu' or 'cuda')
    """
    import matplotlib.pyplot as plt

    model.to(device)  # Move model to the specified device

    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)  # Move data to the specified device

    # Generate reconstructed images
    _, outputs = model(images)

    # Plot the original images and reconstructed images
    fig, axes = plt.subplots(
        nrows=2, ncols=10, sharex=True, sharey=True, figsize=(8, 2)
    )

    for images, row in zip([images, outputs], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.cpu().detach().numpy().squeeze(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


# %%

if __name__ == '__main__':

    # Define hyperparameters
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.001
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
    channels = [6, 16]
    kernel = 3
    initial_layer_side = 28
    inner_dim = 100
    model = ConvAutoencoder().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, num_epochs, criterion, optimizer, device)

    # Test the model
    test_model(model, test_loader, criterion, device)

    # Visualize the results
    visualize_results(model, test_loader, device)
