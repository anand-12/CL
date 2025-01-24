import torch
import torch.nn as nn

# one-hot encoding
def convert_to_network_input(number):
    if number >= 10:
        raise ValueError(f"{number} is too large for one-hot encoding (must be 0-9)")
    result = [0] * 10
    result[number] = 1
    return result

class SimpleAdditionNetwork(nn.Module):
    def __init__(self):
        super(SimpleAdditionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(22, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 11),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

def create_ones_training_examples():
    inputs = []
    correct_answers = []
    
    # Only 1+1 to 9+1
    for num in range(1, 10):
        current_input = convert_to_network_input(num)
        current_input += convert_to_network_input(1)
        current_input += [1, 0]
        inputs.append(current_input)
        correct_answers.append([1 if i == (num + 1) else 0 for i in range(11)])
    
    return torch.FloatTensor(inputs), torch.FloatTensor(correct_answers)

def create_twos_training_examples():
    inputs = []
    correct_answers = []
    
    # Only 1+2 to 9+2
    for num in range(1, 10):
        current_input = convert_to_network_input(num)
        current_input += convert_to_network_input(2)
        current_input += [1, 0]
        inputs.append(current_input)
        correct_answers.append([1 if i == (num + 2) else 0 for i in range(11)])
    
    return torch.FloatTensor(inputs), torch.FloatTensor(correct_answers)

def train_network_phase(network, inputs, correct_answers, epochs=10000):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        outputs = network(inputs)
        loss = loss_function(outputs, correct_answers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

def test_addition(network, num1, num2):
    if num1 >= 10 or num2 >= 10:
        raise ValueError("Numbers must be between 0 and 9")
    
    input_data = convert_to_network_input(num1)
    input_data += convert_to_network_input(num2)
    input_data += [1, 0] 
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    
    with torch.no_grad():
        output = network(input_tensor)
        predicted_num = torch.argmax(output).item()
    return predicted_num

def test_ones_addition(network):
    print("\nTesting additions with 1:")
    correct = 0
    total = 0
    for num in range(1, 10):
        result = test_addition(network, num, 1)
        correct += (result == (num + 1))
        total += 1
        print(f"{num} + 1 = {result} (Should be {num + 1})")
    accuracy = (correct / total) * 100
    print(f"Accuracy on ones addition: {accuracy:.2f}%")
    return accuracy

def test_twos_addition(network):
    print("\nTesting additions with 2:")
    correct = 0
    total = 0
    for num in range(1, 10):
        result = test_addition(network, num, 2)
        correct += (result == (num + 2))
        total += 1
        print(f"{num} + 2 = {result} (Should be {num + 2})")
    accuracy = (correct / total) * 100
    print(f"Accuracy on twos addition: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    network = SimpleAdditionNetwork()

    inputs_ones, answers_ones = create_ones_training_examples()
    train_network_phase(network, inputs_ones, answers_ones)

    accuracy_ones_1 = test_ones_addition(network)

    inputs_twos, answers_twos = create_twos_training_examples()
    train_network_phase(network, inputs_twos, answers_twos)

    accuracy_twos = test_twos_addition(network)

    accuracy_ones_2 = test_ones_addition(network)
    
    if accuracy_ones_2 < accuracy_ones_1:
        print(f"performance on ones dropped by {accuracy_ones_1 - accuracy_ones_2:.2f}%")