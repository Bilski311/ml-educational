import torch.cuda
from torch import nn
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits


model = NeuralNetwork().to(device)
print(model)
X = torch.rand(1, 28, 28, device="cpu")
print(X)
plt.imshow(X.squeeze(), cmap="gray")
X = X.to(device)
print(X.device)
logits = model(X)
predicted_probability = nn.Softmax(dim=1)(logits)
y_pred = predicted_probability.argmax(1)
plt.title(y_pred)
plt.show()

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")