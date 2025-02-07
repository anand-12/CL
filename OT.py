import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
# -----------------------
# Sinkhorn distance function
# -----------------------
def sinkhorn_loss(x, y, epsilon=0.1, max_iter=100, tol=1e-9):
    """
    Computes the entropic regularized Wasserstein (Sinkhorn) distance
    between two empirical distributions represented by samples x and y.
    Args:
        x: Tensor of shape [n, d]
        y: Tensor of shape [m, d]
        epsilon: Regularization coefficient
        max_iter: Maximum Sinkhorn iterations
        tol: Convergence tolerance
    Returns:
        cost: A scalar tensor representing the Sinkhorn cost.
    """
    n, d = x.shape
    m, _ = y.shape
    # Compute squared Euclidean distance matrix (cost matrix)
    C = torch.cdist(x, y, p=2) ** 2  # shape [n, m]

    # Uniform marginals
    mu = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    nu = torch.full((m,), 1.0 / m, device=x.device, dtype=x.dtype)

    # Initialize dual variables
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    K = torch.exp(-C / epsilon)

    for i in range(max_iter):
        u_prev = u.clone()
        u = mu / (K @ v + 1e-8)
        v = nu / (K.transpose(0, 1) @ u + 1e-8)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    gamma = torch.diag(u) @ K @ torch.diag(v)
    cost = torch.sum(gamma * C)
    return cost

# -----------------------
# BayesianLinear with OT divergence method
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
        
        # Prior parameters (initialized as N(0,1))
        self.register_buffer('weight_prior_mu', torch.zeros(out_features, in_features))
        self.register_buffer('weight_prior_log_var', torch.zeros(out_features, in_features))
        self.register_buffer('bias_prior_mu', torch.zeros(out_features))
        self.register_buffer('bias_prior_log_var', torch.zeros(out_features))
        
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
        kl = 0.5 * (log_var_p - log_var_q + 
                    (torch.exp(log_var_q) + (mu_q - mu_p)**2) / torch.exp(log_var_p) - 1)
        return torch.sum(kl)
    
    def update_prior(self):
        self.weight_prior_mu.data.copy_(self.weight_mu.data)
        self.weight_prior_log_var.data.copy_(self.weight_log_var.data)
        self.bias_prior_mu.data.copy_(self.bias_mu.data)
        self.bias_prior_log_var.data.copy_(self.bias_log_var.data)
    
    def sample_weights(self, n_samples=10, use_prior=False):
        """
        Sample n_samples from the weight distribution.
        If use_prior is True, samples from the stored prior are returned.
        """
        if use_prior:
            mu = self.weight_prior_mu
            log_var = self.weight_prior_log_var
        else:
            mu = self.weight_mu
            log_var = self.weight_log_var
        std = torch.exp(0.5 * log_var)
        # Sample n_samples for each weight element; result shape: [n_samples, out_features, in_features]
        eps = torch.randn(n_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps
        # Flatten each sample to a vector
        samples = samples.view(n_samples, -1)
        return samples

    def sample_bias(self, n_samples=10, use_prior=False):
        if use_prior:
            mu = self.bias_prior_mu
            log_var = self.bias_prior_log_var
        else:
            mu = self.bias_mu
            log_var = self.bias_log_var
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(n_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps
        # Flatten each sample to a vector
        samples = samples.view(n_samples, -1)
        return samples

    def ot_divergence(self, n_samples=10, epsilon=0.1, sinkhorn_iters=50):
        """
        Compute the Sinkhorn OT distance between n_samples drawn from the current posterior
        and n_samples drawn from the stored prior for both weights and biases.
        """
        # Sample weights and biases from current distribution
        current_w = self.sample_weights(n_samples=n_samples, use_prior=False)
        current_b = self.sample_bias(n_samples=n_samples, use_prior=False)
        # Sample from the stored prior
        prior_w = self.sample_weights(n_samples=n_samples, use_prior=True)
        prior_b = self.sample_bias(n_samples=n_samples, use_prior=True)
        
        # Compute Sinkhorn loss on weights and biases (we sum them)
        ot_w = sinkhorn_loss(current_w, prior_w, epsilon=epsilon, max_iter=sinkhorn_iters)
        ot_b = sinkhorn_loss(current_b, prior_b, epsilon=epsilon, max_iter=sinkhorn_iters)
        return ot_w + ot_b

# -----------------------
# Bayesian MLP Model with OT divergence summing across layers
# -----------------------
class BayesianMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=200, output_size=10):
        super().__init__()
        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, hidden_size)
        self.fc3 = BayesianLinear(hidden_size, output_size)
        
    def forward(self, x, sample=True):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = self.fc3(x, sample)
        return x
    
    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()
    
    def ot_divergence(self, n_samples=10, epsilon=0.1, sinkhorn_iters=50):
        ot_fc1 = self.fc1.ot_divergence(n_samples=n_samples, epsilon=epsilon, sinkhorn_iters=sinkhorn_iters)
        ot_fc2 = self.fc2.ot_divergence(n_samples=n_samples, epsilon=epsilon, sinkhorn_iters=sinkhorn_iters)
        ot_fc3 = self.fc3.ot_divergence(n_samples=n_samples, epsilon=epsilon, sinkhorn_iters=sinkhorn_iters)
        return ot_fc1 + ot_fc2 + ot_fc3
    
    def update_priors(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.update_prior()

# -----------------------
# Permuted MNIST Data Loader
# -----------------------
def get_permuted_mnist(task_id, permute_seed=42):
    torch.manual_seed(permute_seed + task_id)
    permutation = torch.randperm(784)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# -----------------------
# Training and Evaluation Functions
# -----------------------
def train_task(model, train_loader, optimizer, device, num_epochs=10, ot_weight=0.1, n_ot_samples=10):
    model.train()
    N_task = len(train_loader.dataset)
    total_loss = 0
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, sample=True)
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(output, target)
            # KL divergence term normalized by task size
            kl = model.kl_divergence() / N_task
            # OT regularization computed on each layer
            ot_loss = model.ot_divergence(n_samples=n_ot_samples, epsilon=0.1, sinkhorn_iters=50)
            loss = ce_loss + kl + ot_weight * ot_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{N_task}] Loss: {loss.item():.6f}")
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def evaluate(model, task_id, device, permute_seed=42):
    accuracies = []
    for t in range(task_id + 1):
        _, test_dataset = get_permuted_mnist(t, permute_seed)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, sample=False)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        acc = correct / total
        accuracies.append(acc)
        print(f"Task {t} Accuracy: {100*acc:.2f}%")
    return accuracies

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Bayesian MLP with OT divergence on Permuted MNIST')
    parser.add_argument('--ot_weight', type=float, default=0.9,
                      help='Weight for the OT divergence term (default: 0.9)')
    parser.add_argument('--n_ot_samples', type=int, default=100,
                      help='Number of samples for OT computation (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=200,
                      help='Hidden layer size (default: 200)')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of epochs per task (default: 10)')
    parser.add_argument('--num_tasks', type=int, default=10,
                      help='Number of sequential tasks (default: 10)')
    parser.add_argument('--permute_seed', type=int, default=0,
                      help='Seed for permutation generation (default: 0)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run on (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize model and optimizer
    model = BayesianMLP(hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    all_accuracies = []
    
    for task_id in range(args.num_tasks):
        print(f"\n--- Training Task {task_id} ---")
        train_dataset, _ = get_permuted_mnist(task_id, args.permute_seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        train_loss = train_task(
            model, 
            train_loader, 
            optimizer, 
            device, 
            num_epochs=args.num_epochs, 
            ot_weight=args.ot_weight, 
            n_ot_samples=args.n_ot_samples
        )
        
        print(f"Task {task_id} Training Loss: {train_loss:.4f}")
        # Update the priors after finishing a task
        model.update_priors()
        
        print(f"\n--- Evaluating after Task {task_id} ---")
        accuracies = evaluate(model, task_id, device, args.permute_seed)
        all_accuracies.append(accuracies)
    
    print("\nFinal Accuracies:")
    for task_id, accs in enumerate(all_accuracies):
        print(f"After Task {task_id}: {[f'{a*100:.2f}%' for a in accs]}")

if __name__ == "__main__":
    main()