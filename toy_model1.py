# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# %%
# Simple feedforward neural network with one hidden layer
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleModel, self).__init__()
        self.W = nn.Parameter(
            torch.randn(hidden_dim, input_dim, requires_grad=True) / 5
        )  # Adjusted the shape
        self.b = nn.Parameter(torch.randn(input_dim, 1, requires_grad=True))

    def forward(self, x):
        x1 = self.W @ x.t()
        x2 = self.W.t() @ x1 + self.b
        x3 = torch.relu(x2)
        return x3.t()


# %%
# loss function
def loss_function(output, target):
    weight = torch.tensor(
        [0.7**i for i in range(target.shape[1])], requires_grad=False
    )
    output = torch.sum(weight * (output - target) ** 2, dim=1)
    return torch.mean(output)
    # return torch.mean(weight*(output-target)**2)


# %%
# Create the model
input_dim = 20
hidden_dim = 5
model = SimpleModel(input_dim, hidden_dim)

# Define a loss function and optimizer
# criterion = nn.MSELoss()  # Mean Squared Error for demonstration
criterion = loss_function
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy data for demonstration
x = torch.randn(1024, input_dim)  # Batch of 32 samples with 20 features each
# y = torch.randn(32, hidden_dim)  # Random target values for demonstration


# dropout mask
def dropout_mask(tensor, p=0.5):
    """Set entries in a tensor to 0 with probability p."""
    mask = (torch.rand(tensor.size()) > p).float()
    return tensor * mask


# %%
# model system
W = torch.randn(hidden_dim, input_dim)
b = torch.randn(input_dim, 1)
x1 = W @ x.t()
x2 = W.t() @ x1 + b
x3 = torch.relu(x2)
x4 = x3.t()
weight = torch.tensor([0.7**i for i in range(x.shape[1])])
output = torch.sum(weight * (x4 - x) ** 2, dim=1)
torch.mean(output)


# %%
# Training loop
epochs = 100000
for epoch in range(epochs):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training Complete!")

# %%


# visualize W^T W
def visualize_matrix_times_transpose(matrix):
    # Calculate the product of the matrix and its transpose
    product = np.dot(matrix.T, matrix)

    # Plot the heatmap
    plt.imshow(product, cmap="viridis")
    plt.colorbar()
    plt.title("Matrix times its Transpose")
    plt.show()


# Example usage:
matrix = np.array([[1, 2], [3, 4], [5, 6]])
visualize_matrix_times_transpose(matrix)
# %%
visualize_matrix_times_transpose(model.W.detach().numpy())
