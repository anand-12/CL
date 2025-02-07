import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. Deep Gaussian Process Layer ===============================================
class DGP_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, num_inducing=50):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing

        # Inducing points (Z)
        self.Z = nn.Parameter(torch.randn(num_inducing, input_dim))
        
        # Variational distribution q(U) ~ N(m, S)
        self.q_m = nn.Parameter(torch.randn(num_inducing, output_dim))
        self.q_L = nn.Parameter(torch.eye(num_inducing))  # Cholesky factor
        
        # Prior parameters (initialized as N(0, I))
        self.register_buffer('prior_m', torch.zeros_like(self.q_m))
        self.register_buffer('prior_L', torch.eye(num_inducing))
        
        # Kernel parameters
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))
        self.log_variance = nn.Parameter(torch.tensor(0.0))

    def kernel(self, X1, X2):
        ls = torch.exp(self.log_lengthscale)
        var = torch.exp(self.log_variance)
        dist = torch.cdist(X1/ls, X2/ls)
        return var * torch.exp(-0.5 * dist.pow(2))

    def forward(self, X, sample=True):
        K_zz = self.kernel(self.Z, self.Z) + 1e-6*torch.eye(self.num_inducing)
        K_xz = self.kernel(X, self.Z)
        
        # Cholesky decomposition
        L_z = torch.linalg.cholesky(K_zz)
        alpha = torch.cholesky_solve(K_xz.mT, L_z).mT  # K_xz K_zz^{-1}
        
        # Mean and covariance
        mean = alpha @ self.q_m
        cov_term = alpha @ (self.q_L @ self.q_L.mT - K_zz) @ alpha.mT
        cov = self.kernel(X, X) + cov_term + 1e-6*torch.eye(X.size(0))
        
        if sample:
            return td.MultivariateNormal(mean, covariance_matrix=cov).rsample()
        return mean

    def kl_divergence(self):
        p = td.MultivariateNormal(self.prior_m, scale_tril=self.prior_L)
        q = td.MultivariateNormal(self.q_m, scale_tril=self.q_L)
        return td.kl_divergence(q, p)
    
    def update_prior(self):
        self.prior_m.data.copy_(self.q_m.data)
        self.prior_L.data.copy_(self.q_L.data)

# 2. Deep GP Model =============================================================
class VCL_DGP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[100, 50], 
                 num_inducing=50, activation=torch.tanh):
        super().__init__()
        self.activation = activation
        
        # Create DGP layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            DGP_Layer(dims[i], dims[i+1], num_inducing) 
            for i in range(len(dims)-1)
        ])
        
    def forward(self, X, sample=True):
        h = X
        for layer in self.layers:
            h = layer(h, sample=sample)
            h = self.activation(h)
        return h
    
    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def update_priors(self):
        for layer in self.layers:
            layer.update_prior()

# 3. Synthetic Task Generator ==================================================
class RotatedMNIST(Dataset):
    def __init__(self, task_id, num_samples=1000):
        self.task_id = task_id
        self.num_samples = num_samples
        self.rotation = task_id * 15  # 15 degree increments
        self.data = torch.randn(num_samples, 784)
        self.labels = torch.randint(0, 10, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.data[idx].view(28, 28)
        img = torch.tensor(np.rot90(img.numpy(), self.task_id).copy())
        return img.flatten(), self.labels[idx]

# 4. Training & Evaluation =====================================================
def train_task(model, train_loader, optimizer, device, task_id):
    model.train()
    N = len(train_loader.dataset)
    
    for epoch in range(10):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, y)
            
            # KL divergence
            kl = model.kl_divergence()
            
            # ELBO loss
            loss = ce_loss + kl / N
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Task {task_id} | Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

def evaluate(model, tasks_seen, device):
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for task_id in range(tasks_seen):
            dataset = RotatedMNIST(task_id)
            loader = DataLoader(dataset, batch_size=100, shuffle=False)
            correct = 0
            total = 0
            
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X, sample=False)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
            acc = correct / total
            accuracies.append(acc)
            print(f"Task {task_id} Accuracy: {acc*100:.2f}%")
    
    return accuracies

# 5. Main Execution ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 5
    batch_size = 128
    
    # Initialize model
    model = VCL_DGP(input_dim=784, output_dim=10, 
                   hidden_dims=[200, 100], num_inducing=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Continual learning loop
    all_accuracies = []
    for task_id in range(num_tasks):
        # Train on current task
        train_dataset = RotatedMNIST(task_id)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_task(model, train_loader, optimizer, device, task_id)
        
        # Update priors for next task
        model.update_priors()
        
        # Evaluate on all previous tasks
        print(f"\nAfter Task {task_id} Evaluation:")
        accuracies = evaluate(model, task_id+1, device)
        all_accuracies.append(accuracies)
    
    # Print final results
    print("\nFinal Task Accuracies Matrix:")
    for i, accs in enumerate(all_accuracies):
        print(f"After Task {i}: {[f'{a*100:.1f}%' for a in accs]}")

if __name__ == "__main__":
    main()