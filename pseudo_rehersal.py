import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import seaborn as sns

class MNISTDigitPairs(Dataset):
    def __init__(self, pair_index, train=True):
        """
        pair_index: 0 for (0,1), 1 for (2,3), 2 for (4,5), 3 for (6,7), 4 for (8,9)
        """
        # Load MNIST
        self.mnist = torchvision.datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
        
        # Calculate the digit pairs
        self.digit1 = pair_index * 2
        self.digit2 = pair_index * 2 + 1
        
        # Filter only the digits we want
        self.indices = []
        for idx, (_, label) in enumerate(self.mnist):
            if label in [self.digit1, self.digit2]:
                self.indices.append(idx)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.mnist[self.indices[idx]]
        # Convert to binary classification (first digit is 1, second is 0)
        binary_label = 1.0 if label == self.digit1 else 0.0
        return image, torch.tensor([binary_label], dtype=torch.float32)

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total

def evaluate_all_tasks(model, test_loaders, device):
    """Evaluate model on all tasks"""
    accuracies = []
    for loader in test_loaders:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracies.append(correct / total)
    
    return accuracies

def demonstrate_sequential_learning():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleConvNet().to(device)
    
    # Training parameters
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001
    num_pairs = 5  # 5 pairs of digits (0-1, 2-3, 4-5, 6-7, 8-9)
    
    # Create dataloaders for all pairs
    train_loaders = []
    test_loaders = []
    
    for pair_idx in range(num_pairs):
        train_loader = DataLoader(
            MNISTDigitPairs(pair_idx, train=True),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            MNISTDigitPairs(pair_idx, train=False),
            batch_size=batch_size
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Matrix to store accuracies after each task
    # Rows: current task being learned
    # Columns: performance on all tasks
    accuracy_matrix = np.zeros((num_pairs, num_pairs))
    
    # Train on each pair sequentially
    for current_task in range(num_pairs):
        print(f"\nTraining on Task {current_task} "
              f"(Digits {current_task*2} vs {current_task*2 + 1})...")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Train on current task
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model, train_loaders[current_task],
                optimizer, criterion, device
            )
            
            # Evaluate on all tasks
            task_accuracies = evaluate_all_tasks(model, test_loaders, device)
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            for task_idx, acc in enumerate(task_accuracies):
                print(f"Task {task_idx} Acc: {acc:.4f}")
        
        # Store final accuracies for this training phase
        accuracy_matrix[current_task] = evaluate_all_tasks(model, test_loaders, device)
    
    # Visualize the results
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=[f'{i*2}-{i*2+1}' for i in range(num_pairs)],
                yticklabels=[f'{i*2}-{i*2+1}' for i in range(num_pairs)])
    plt.xlabel('Task (Digit Pair)')
    plt.ylabel('After Training on Task')
    plt.title('Task Performance Matrix\nAccuracy on each task (columns) after training on task (rows)')
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\nAnalysis of Forgetting:")
    for task in range(num_pairs):
        initial_acc = accuracy_matrix[task, task]  # Accuracy right after learning
        final_acc = accuracy_matrix[-1, task]      # Accuracy after learning all tasks
        print(f"\nTask {task} (Digits {task*2}-{task*2+1}):")
        print(f"Initial accuracy: {initial_acc:.4f}")
        print(f"Final accuracy: {final_acc:.4f}")
        print(f"Performance drop: {initial_acc - final_acc:.4f}")

if __name__ == "__main__":
    demonstrate_sequential_learning()