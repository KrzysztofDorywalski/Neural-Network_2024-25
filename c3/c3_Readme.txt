Usefull chunks of code:


    # Print gradients for each parameter
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient for {name}: {param.grad}')
        else:
            print(f'{name} has no gradient')



# Loss function: Binary Cross Entropy
criterion = nn.BCELoss()


class XORNeuralNetwork(nn.Module):
    def __init__(self):
        super(XORNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 2)  # Hidden layer with 2 neurons
        self.layer2 = nn.Linear(2, 1)  # Output layer
        self.relu = nn.ReLU()          # Activation function for non-linearity

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Hidden layer + ReLU
        x = torch.sigmoid(self.layer2(x))  # Output layer + Sigmoid
        return x
