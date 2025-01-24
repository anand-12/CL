import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import torch
import torch.nn as nn

def convert_to_network_input(number):
    if number >= 10:
        raise ValueError(f"{number} is too large for one-hot encoding (must be 0-9)")
    result = [0] * 10
    result[number] = 1
    return result

class GPAdditionLearner:
    def __init__(self):
        # Using RBF kernel with optimizable length scale
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.model = GaussianProcessRegressor(kernel=kernel)
        self.trained_data = []
        
    def create_gp_input(self, num1, num2):
        # Simpler input representation for GP - just use the numbers directly
        return np.array([[num1, num2]])
    
    def train(self, num1, num2, result):
        X = self.create_gp_input(num1, num2)
        y = np.array([result])
        
        # Accumulate training data
        if not hasattr(self, 'X_train'):
            self.X_train = X
            self.y_train = y
        else:
            self.X_train = np.vstack((self.X_train, X))
            self.y_train = np.concatenate((self.y_train, y))
        
        # Retrain on all accumulated data
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, num1, num2):
        X = self.create_gp_input(num1, num2)
        prediction = self.model.predict(X)
        return round(prediction[0])

def test_addition_gp(gp, num_range, add_with):
    print(f"\nTesting additions with {add_with}:")
    correct = 0
    total = 0
    for num in range(1, num_range):
        result = gp.predict(num, add_with)
        correct += (result == (num + add_with))
        total += 1
        print(f"{num} + {add_with} = {result} (Should be {num + add_with})")
    accuracy = (correct / total) * 100
    print(f"Accuracy on {add_with}s addition: {accuracy:.2f}%")
    return accuracy

def main():

    gp = GPAdditionLearner()
    
    # Train on adding ones
    print("\nTraining on adding ones...")
    for num in range(1, 10):
        gp.train(num, 1, num + 1)
    
    # Test adding ones
    accuracy_ones_1_gp = test_addition_gp(gp, 10, 1)
    
    # Train on adding twos
    print("\nTraining on adding twos...")
    for num in range(1, 10):
        gp.train(num, 2, num + 2)
    
    # Test both
    accuracy_twos_gp = test_addition_gp(gp, 10, 2)
    accuracy_ones_2_gp = test_addition_gp(gp, 10, 1)
    
    if accuracy_ones_2_gp < accuracy_ones_1_gp:
        print(f"\nGP - Catastrophic forgetting observed: Performance on ones dropped by {accuracy_ones_1_gp - accuracy_ones_2_gp:.2f}%")
    else:
        print(f"Original accuracy on ones: {accuracy_ones_1_gp:.2f}%")
        print(f"Final accuracy on ones: {accuracy_ones_2_gp:.2f}%")

if __name__ == "__main__":
    main()