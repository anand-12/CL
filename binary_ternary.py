import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

def generate_single_sample(iteration):
    if iteration >= 30:
        chosen_class = np.random.choice([0, 1, 2])
    else:
        chosen_class = np.random.choice([0, 1])
    if chosen_class == 0:
        x = np.random.normal(-1, 1, 1)
    elif chosen_class == 1:
        x = np.random.normal(1, 1, 1)
    else:
        x = np.random.normal(0, 1, 1)
    return x.reshape(-1, 1), np.array([chosen_class])

class OnlineGPC:
    def __init__(self):
        self.kernel = RBF(length_scale=1.0)
        self.model = GaussianProcessClassifier(kernel=self.kernel)
        self.X = None
        self.y = None
        self.initialized = False
        self.n_classes = 2

    def initialize(self):
        X0, y0 = generate_single_sample(iteration=0)
        while y0[0] != 0:
            X0, y0 = generate_single_sample(iteration=0)
        X1, y1 = generate_single_sample(iteration=0)
        while y1[0] != 1:
            X1, y1 = generate_single_sample(iteration=0)
        self.X = np.vstack((X0, X1))
        self.y = np.hstack((y0, y1))
        self.model.fit(self.X, self.y)
        self.initialized = True

    def update(self, X_new, y_new, iteration):
        if not self.initialized:
            self.initialize()
            return
        self.n_classes = max(self.n_classes, y_new[0] + 1)
        self.X = np.vstack((self.X, X_new))
        self.y = np.hstack((self.y, y_new))
        self.model.fit(self.X, self.y)

    def predict(self, X):
        if not self.initialized:
            return np.array([[1/self.n_classes] * self.n_classes] * len(X))
        return self.model.predict_proba(X)

# Setup
n_iterations = 500
x_test = np.linspace(-6, 6, 100).reshape(-1, 1)
gpc = OnlineGPC()

# Train the model
for i in range(n_iterations):
    X_new, y_new = generate_single_sample(iteration=i)
    gpc.update(X_new, y_new, i)

# Plot true distributions and final learned distribution
plt.figure(figsize=(12, 6))

# True distributions
x_plot = np.linspace(-6, 6, 1000)
plt.subplot(2, 1, 1)
plt.title('True Distributions')


# Final distribution (after 30 iterations)
plt.plot(x_plot, 1/3 * np.exp(-(x_plot + 1)**2 / 2) / np.sqrt(2*np.pi), 
         'r-', label='Class 1 (Final)')
plt.plot(x_plot, 1/3 * np.exp(-(x_plot - 1)**2 / 2) / np.sqrt(2*np.pi), 
         'b-', label='Class 2 (Final)')
plt.plot(x_plot, 1/3 * np.exp(-(x_plot)**2 / 2) / np.sqrt(2*np.pi), 
         'g-', label='Class 3 (Final)')
plt.legend()

# Learned distribution
plt.subplot(2, 1, 2)
plt.title('Final Learned Distribution')
y_pred = gpc.predict(x_test)
colors = ['red', 'blue', 'green']
class_labels = ['Class 1', 'Class 2', 'Class 3']
for class_idx in range(gpc.n_classes):
    plt.plot(x_test, y_pred[:, class_idx], 
             c=colors[class_idx], label=class_labels[class_idx],
             linestyle='-', alpha=0.7)
plt.ylim(-0.1, 1.1)
plt.ylabel('Probability')
plt.xlabel('x')
plt.legend()

plt.tight_layout()
plt.show()