import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

def get_dataloaders(batch_size = 64):
    """
    Creates CIFAR-10 train and test DataLoaders.

    Args:
        batch_size (int): Number of samples per batch. Default is 64.
        
    Returns:
        tuple: (train_loader, test_loader)
            - train_loader (DataLoader): DataLoader for training set.
            - test_loader (DataLoader): DataLoader for test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize RGB
    ])
    
    train_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = transform)
    test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = transform)
    
    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """
    Trains the given model on the provided training data.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function (e.g. CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g. Adam, SGD).
        epochs (int): Number of training epochs.
        device (torch.device): Device to run training on (CPU or GPU).
        
    Returns:
        None
    """
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def evaluate_model(model, test_loader, device):
    """
    Evaluates the given model on the provided test dataset and prints accuracy.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for test dataset.
        device (torch.device): Device to run evaluation on (CPU or GPU).
        
    Returns:
        None
    """
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on CIFAR-10 test set: {100 * correct / total:.2f}%")

def main():
    """
    Main entry point: sets up data, model, loss, optimizer,
    trains the model and evaluates its performance.

    Args:
        None
        
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders()
    model = TinyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train_model(model, train_loader, criterion, optimizer, epochs = 5, device = device)
    evaluate_model(model, test_loader, device)
    
if __name__ == "__main__":
    main()