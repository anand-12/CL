import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    
    def update_priors(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.update_prior()

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

def train_task(model, train_loader, optimizer, device, num_epochs=10):
    model.train()
    N_task = len(train_loader.dataset)
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, sample=True)
            ce_loss = F.cross_entropy(output, target)
            kl = model.kl_divergence()
            loss = ce_loss + kl / N_task
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{N_task}] Loss: {loss.item():.6f}')

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
        print(f'Task {t} Accuracy: {100*acc:.2f}%')
    return accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tasks = 5
    hidden_size = 200
    batch_size = 256
    lr = 1e-3
    num_epochs = 10
    permute_seed = 42
    
    model = BayesianMLP(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    all_accuracies = []
    
    for task_id in range(num_tasks):
        print(f"\n--- Training Task {task_id} ---")
        train_dataset, _ = get_permuted_mnist(task_id, permute_seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_task(model, train_loader, optimizer, device, num_epochs)
        model.update_priors()
        print(f"\n--- Evaluating after Task {task_id} ---")
        accuracies = evaluate(model, task_id, device, permute_seed)
        all_accuracies.append(accuracies)
    
    print("\nFinal Accuracies:")
    for task_id, accs in enumerate(all_accuracies):
        print(f"After Task {task_id}: {[f'{a*100:.2f}%' for a in accs]}")

if __name__ == "__main__":
    main()
