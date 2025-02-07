import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import os

# -----------------------
# Sliced Wasserstein Distance (SWD)
# -----------------------
def sliced_wasserstein(x, y, n_proj=200, p=2):
    device = x.device
    n_features = x.size(1)
    
    # Generate random projection directions
    proj = torch.randn(n_features, n_proj, device=device)
    proj = proj / torch.norm(proj, dim=0, keepdim=True)
    
    # Project and sort
    x_proj = x @ proj
    y_proj = y @ proj
    x_proj_sorted = torch.sort(x_proj, dim=0)[0]
    y_proj_sorted = torch.sort(y_proj, dim=0)[0]
    
    # Compute SWD
    delta = x_proj_sorted - y_proj_sorted
    return torch.mean(torch.abs(delta)**p)

# -----------------------
# Bayesian Linear Layer
# -----------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        
        # Prior parameters
        self.register_buffer('weight_prior_mu', torch.zeros_like(self.weight_mu))
        self.register_buffer('weight_prior_log_var', torch.zeros_like(self.weight_log_var))
        self.register_buffer('bias_prior_mu', torch.zeros_like(self.bias_mu))
        self.register_buffer('bias_prior_log_var', torch.zeros_like(self.bias_log_var))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -6)
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_log_var, -6)
    
    def forward(self, x, sample=True):
        if self.training or sample:
            weight = self.weight_mu + torch.exp(0.5 * self.weight_log_var) * torch.randn_like(self.weight_log_var)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_log_var) * torch.randn_like(self.bias_log_var)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        kl_weight = self._kl_gaussian(self.weight_mu, self.weight_log_var, 
                                     self.weight_prior_mu, self.weight_prior_log_var)
        kl_bias = self._kl_gaussian(self.bias_mu, self.bias_log_var,
                                   self.bias_prior_mu, self.bias_prior_log_var)
        return kl_weight + kl_bias
    
    def _kl_gaussian(self, mu_q, log_var_q, mu_p, log_var_p):
        return 0.5 * torch.sum(log_var_p - log_var_q + 
                              (torch.exp(log_var_q) + (mu_q - mu_p)**2) / torch.exp(log_var_p) - 1)
    
    def update_prior(self):
        self.weight_prior_mu.data.copy_(self.weight_mu.data)
        self.weight_prior_log_var.data.copy_(self.weight_log_var.data)
        self.bias_prior_mu.data.copy_(self.bias_mu.data)
        self.bias_prior_log_var.data.copy_(self.bias_log_var.data)
    
    def swd_divergence(self, n_samples=50, n_proj=200):
        def sample_params(use_prior=False):
            if use_prior:
                w_mu = self.weight_prior_mu
                w_log_var = self.weight_prior_log_var
                b_mu = self.bias_prior_mu
                b_log_var = self.bias_prior_log_var
            else:
                w_mu = self.weight_mu
                w_log_var = self.weight_log_var
                b_mu = self.bias_mu
                b_log_var = self.bias_log_var
            
            device = w_mu.device
            w_std = torch.exp(0.5 * w_log_var)
            w_eps = torch.randn(n_samples, *w_mu.shape, device=device)
            w = w_mu.unsqueeze(0) + w_std.unsqueeze(0) * w_eps
            
            b_std = torch.exp(0.5 * b_log_var)
            b_eps = torch.randn(n_samples, *b_mu.shape, device=device)
            b = b_mu.unsqueeze(0) + b_std.unsqueeze(0) * b_eps
            
            return w.view(n_samples, -1), b.view(n_samples, -1)
        
        current_w, current_b = sample_params(use_prior=False)
        prior_w, prior_b = sample_params(use_prior=True)
        
        return (sliced_wasserstein(current_w, prior_w, n_proj=n_proj) +
                sliced_wasserstein(current_b, prior_b, n_proj=n_proj))

# -----------------------
# Deep Bayesian MLP
# -----------------------
class DeepBayesianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc4 = BayesianLinear(hidden_dim, output_dim)
    
    def forward(self, x, sample=True):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = F.relu(self.fc3(x, sample))
        x = self.fc4(x, sample)
        return x
    
    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in [self.fc1, self.fc2, self.fc3, self.fc4])
    
    def swd_divergence(self, n_samples=50, n_proj=200):
        return sum(layer.swd_divergence(n_samples, n_proj) for layer in [self.fc1, self.fc2, self.fc3, self.fc4])
    
    def update_priors(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            layer.update_prior()

# -----------------------
# Task-Specific Data Loaders
# -----------------------
def get_cifar10_split(task_id):

    class RemappedSubset(Dataset):
        def __init__(self, subset, start_class):
            self.subset = subset
            self.start_class = start_class

        def __getitem__(self, idx):
            data, target = self.subset[idx]
            return data, target - self.start_class

        def __len__(self):
            return len(self.subset)

    classes_per_task = 2
    start_class = task_id * classes_per_task
    end_class = start_class + classes_per_task
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_indices = [i for i, target in enumerate(full_train_dataset.targets) if start_class <= target < end_class]
    test_indices = [i for i, target in enumerate(full_test_dataset.targets) if start_class <= target < end_class]
    
    train_dataset = RemappedSubset(Subset(full_train_dataset, train_indices), start_class)
    test_dataset = RemappedSubset(Subset(full_test_dataset, test_indices), start_class)
    
    return train_dataset, test_dataset

def get_rotated_cifar10(task_id, rotation_seed=42):
    torch.manual_seed(rotation_seed + task_id)
    angle = task_id * 15  # 15° increments per task (0°, 15°, 30°, ...)
    
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(angle, angle)),  # Fixed rotation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# -----------------------
# Training and Evaluation
# -----------------------
def train_task(model, train_loader, optimizer, device, num_epochs=10, use_swd=False, swd_weight=1.0, n_swd_samples=50, n_proj=200):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data, sample=True)
            ce_loss = F.cross_entropy(output, target)
            kl = model.kl_divergence() / len(train_loader.dataset)
            loss = ce_loss + kl
            
            if use_swd:
                swd = model.swd_divergence(n_samples=n_swd_samples, n_proj=n_proj)
                loss += swd_weight * swd
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

def evaluate(model, task_id, device, task_type="cifar10_split"):
    model.eval()
    accuracies = []
    for t in range(task_id + 1):
        if task_type == "cifar10_split":
            _, test_dataset = get_cifar10_split(t)
        elif task_type == "rotated_cifar10":
            _, test_dataset = get_rotated_cifar10(t)
        else:
            raise ValueError("Unsupported task type")
        
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, sample=False)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        acc = correct / len(test_dataset)
        accuracies.append(acc)
        print(f"Task {t} Accuracy: {100*acc:.2f}%")
    return accuracies

# -----------------------
# Main Function
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Continual Learning with Bayesian Deep MLP')
    parser.add_argument('--task_type', type=str, default='cifar10_split', choices=['cifar10_split', 'rotated_cifar10'], help='Type of task')
    parser.add_argument('--use_swd', action='store_true', help='Use SWD regularization')
    parser.add_argument('--swd_weight', type=float, default=1.0, help='Weight for SWD term')
    parser.add_argument('--n_swd_samples', type=int, default=50, help='Number of samples for SWD')
    parser.add_argument('--n_proj', type=int, default=200, help='Number of projections for SWD')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs per task')
    parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.task_type == "cifar10_split":
        input_dim = 3 * 32 * 32
        output_dim = 2  # Each task has 2 classes
    elif args.task_type == "rotated_cifar10":
        input_dim = 3 * 32 * 32
        output_dim = 10
    else:
        raise ValueError("Unsupported task type")
    
    hidden_dim = 512
    model = DeepBayesianMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    all_accuracies = []
    for task_id in range(args.num_tasks):
        print(f"\n=== Training Task {task_id} ===")
        if args.task_type == "cifar10_split":
            train_dataset, _ = get_cifar10_split(task_id)
        elif args.task_type == "rotated_cifar10":
            train_dataset, _ = get_rotated_cifar10(task_id)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        train_task(model, train_loader, optimizer, device,
                   num_epochs=args.num_epochs,
                   use_swd=args.use_swd,
                   swd_weight=args.swd_weight,
                   n_swd_samples=args.n_swd_samples,
                   n_proj=args.n_proj)
        
        model.update_priors()
        
        print(f"\n=== Evaluation after Task {task_id} ===")
        accuracies = evaluate(model, task_id, device, task_type=args.task_type)
        all_accuracies.append(accuracies)
    
    print("\nFinal Accuracies Matrix:")
    for task_id, accs in enumerate(all_accuracies):
        print(f"After Task {task_id}: {[f'{a*100:.1f}%' for a in accs]}")

if __name__ == "__main__":
    main()