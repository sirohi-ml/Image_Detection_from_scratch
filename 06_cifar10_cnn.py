### The idea of this file is to understand what changes when we move from "Greyscale" images to "RGB" images.
### We are trying to predict images that carry 10 distinct labels like "Airplanes", "Cats".... etc

import torch
from torchvision import datasets, transforms

## Convert images into PyTorch tensors
transform = transforms.ToTensor()


## Load the CIFAR-10 dataset

train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True, 
    transform = transform
)

test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True, 
    transform = transform
)

print("Number of training images:", len(train_dataset))
print("Number of testing images:", len(test_dataset))

## Examining one image
image, label = train_dataset[0]

red = image[0]
green = image[1]
blue = image[2]

print("Red shape:", red.shape)
print("Green shape:", green.shape)
print("Blue shape:", blue.shape)

print("Red first pixel:", red[0, 0])
print("Green first pixel:", green[0, 0])
print("Blue first pixel:", blue[0, 0])


## Data Loader to for batching 
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset, 
    batch_size = 64, 
    shuffle = True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size = 64, 
    shuffle = False
)

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)             # we should see [64, 3, 32, 32]
print("Labels shape:", labels.shape)       # [64] 




## Building the convolution model
import torch.nn as nn

class CIFAR10Cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 10,
            kernel_size = 3
        )

        self.conv2 = nn.Conv2d(
            in_channels = 10,
            out_channels = 16,
            kernel_size = 3
        )

        self.pool = nn.MaxPool2d(
            stride = 2,
            kernel_size=2
        )

        self.fc = nn.Linear(
            16 * 6 * 6 ,
            10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x

## Model Object

model = CIFAR10Cnn()

images, labels = next(iter(train_loader))

output = model(images)

print("Input shape:", images.shape)
print("Output shape:", output.shape)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)

## Training Loop
best_loss = float("inf")
best_epoch = 0


for epoch in range(100):

    total_loss = 0
    count = 0

    for images, labels in train_loader:

        count += 1

        output = model(images)

        loss = criterion(output, labels)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss/count

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            "best_cifar10_model.pth"
        )

    if epoch % 2 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Loss: {total_loss/count} "
        )


## Loading saved model
model.load_state_dict(
    torch.load("best_cifar10_model.pth")
)

print("Best Epoch:", best_epoch)
print("Best training loss:", best_loss)

## Evaluating model performance
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        output = model(images)

        predictions = output.argmax(1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = correct/total

print("Test accuracy:", accuracy)
print("Test accuracy (%):", accuracy * 100)

