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
        x = np.random.normal(-2, 1, 1)    
    elif chosen_class == 1:
        x = np.random.normal(2, 1, 1)     
    else:
        x = np.random.normal(0, 1, 1)     
    
    return x.reshape(-1, 1), np.array([chosen_class])

class InducingPointsGPC:
    def __init__(self, n_inducing=10):
        self.kernel = RBF(length_scale=1.0)
        self.model = GaussianProcessClassifier(kernel=self.kernel)
        self.X = None
        self.y = None
        self.n_classes = 2
        self.n_inducing = n_inducing
        self.inducing_points = None
        self.inducing_labels = None
        self.initialized = False
        
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
    
    def select_inducing_points(self):
        
        inducing_X = []
        inducing_y = []
        
        
        for class_idx in range(2):
            class_mask = self.y == class_idx
            class_points = self.X[class_mask]
            
            if len(class_points) > 0:
                
                selected_indices = np.linspace(0, len(class_points)-1, 
                                            self.n_inducing//2, dtype=int)
                inducing_X.extend(class_points[selected_indices])
                inducing_y.extend([class_idx] * (self.n_inducing//2))
                
        return np.array(inducing_X), np.array(inducing_y)
        
    def update(self, X_new, y_new, iteration):
        if not self.initialized:
            self.initialize()
            return
            
        if iteration == 30:  
            
            self.inducing_points, self.inducing_labels = self.select_inducing_points()
            print(f"Selected {len(self.inducing_points)} inducing points to preserve Task 1 knowledge")
            
            
            self.X = np.vstack((self.inducing_points, X_new))
            self.y = np.hstack((self.inducing_labels, y_new))
            self.n_classes = 3
        else:
            self.X = np.vstack((self.X, X_new))
            self.y = np.hstack((self.y, y_new))
        
        
        unique_classes = np.unique(self.y)
        if len(unique_classes) >= 2:
            self.model.fit(self.X, self.y)
    
    def predict(self, X):
        if not self.initialized or len(np.unique(self.y)) < 2:
            return np.array([[1/self.n_classes] * self.n_classes] * len(X))
            
        base_probs = self.model.predict_proba(X)
        
        
        if self.n_classes == 3 and base_probs.shape[1] == 2:
            padded_probs = np.zeros((len(X), 3))
            padded_probs[:, :2] = base_probs
            return padded_probs
            
        return base_probs

def run_inducing_points():
    # Setup parameters
    n_iterations = 100
    x_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    gpc = InducingPointsGPC(n_inducing=10)
    colors = ['red', 'blue', 'green']
    class_labels = ['Class 1', 'Class 2', 'Class 3']

    # Run all iterations first to train the model
    for i in range(n_iterations):
        X_new, y_new = generate_single_sample(iteration=i)
        gpc.update(X_new, y_new, i)
    
    # Create the final visualization
    plt.figure(figsize=(12, 6))
    
    # Top subplot: Data points and inducing points
    plt.subplot(2, 1, 1)
    
    # Plot inducing points with star markers
    if gpc.inducing_points is not None:
        for class_idx in range(2):
            mask = gpc.inducing_labels == class_idx
            plt.scatter(gpc.inducing_points[mask], 
                       np.zeros_like(gpc.inducing_points[mask]), 
                       c=colors[class_idx], marker='*', s=200, 
                       label=f'Inducing {class_labels[class_idx]}')
    
    # Plot all data points
    for class_idx in range(gpc.n_classes):
        mask = gpc.y == class_idx
        if np.any(mask):
            plt.scatter(gpc.X[mask], np.zeros_like(gpc.X[mask]), 
                       c=colors[class_idx], label=class_labels[class_idx])
    plt.ylim(-1, 1)
    plt.title('Inducing Points - Final State')
    plt.legend()
    
    # Bottom subplot: Probability distributions
    plt.subplot(2, 1, 2)
    y_pred = gpc.predict(x_test)
    for class_idx in range(gpc.n_classes):
        plt.plot(x_test, y_pred[:, class_idx], 
                c=colors[class_idx], label=f'P(class {class_idx})',
                linestyle='-', alpha=0.7)
    
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Probability')
    plt.xlabel('x')
    plt.legend()
    plt.title('Class Probabilities')
    plt.tight_layout()
    
    # Save the final plot as a high-resolution image
    plt.savefig('final_inducing_points.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_inducing_points()