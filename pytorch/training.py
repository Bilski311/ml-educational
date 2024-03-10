import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

from neural_network import NeuralNetwork


def to_one_hot_encoding(y):
    return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


def train_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        prediction = model(X)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_function):
    model.eval()
    size = len(dataloader.dataset)
    number_of_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            prediction = model(X)
            test_loss += loss_function(prediction, y).item()
            correct += (prediction.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= number_of_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


learning_rate = 1e-3
batch_size = 64
epochs = 20

training_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
    target_transform=Lambda(to_one_hot_encoding))

test_data = datasets.FashionMNIST(
    root="data",
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(to_one_hot_encoding)
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}\n-------------------")
    train_loop(train_dataloader, model, loss_function, optimizer)
    test_loop(test_dataloader, model, loss_function)
print("Done!")