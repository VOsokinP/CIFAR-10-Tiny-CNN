import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#CIFAR-10 Dataset & Loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize RGB
])

train_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = transform)
test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)


#Tiny CNN model
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #in_features = channel * height * width. 8 * 8 because pooling ops half the size 32->16->8
        # self.fc1 = nn.Linear(in_features = 32 * 8 * 8, out_features = 128)
        # self.fc2 = nn.Linear(128, 10) #CIFAR has 10 classes
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # x = x.view(-1, 32 * 8 * 8)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

#Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)

learning_rate = 0.001
epochs = 5

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

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

#Evaluation
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