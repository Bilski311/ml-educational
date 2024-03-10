import random

import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


from training import to_one_hot_encoding

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

model = torch.load('model.pth')
test_data = datasets.FashionMNIST(
    root='data',
    download=True,
    train=False,
    transform=ToTensor()
)

random_index = random.randint(0, len(test_data) - 1)
image, target = test_data[random_index]
target_label = labels_map[target]
prediction = model(image)
predicted_label = labels_map[prediction.argmax(1).item()]
numpy_image = image.squeeze().numpy()
plt.imshow(numpy_image, cmap='gray')
plt.title(f'Predicted: {predicted_label} vs Target: {target_label}')
plt.show()