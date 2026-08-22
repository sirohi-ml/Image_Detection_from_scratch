import torch 

## Lets get the dataset first
## This is the MNIST data used by Yann Lecunn
from torchvision import datasets, transforms

transform = transforms.ToTensor()                  # converts each MNIST image into a tensor

train_dataset = datasets.MNIST(
    root = "./data",
    train = True, 
    download = True, 
    transform = transform
)

test_dataset = datasets.MNIST(
    root = "./data",
    train = False,
    download = True,
    transform = transform
)

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)


## Introduction to dataloader

from torch.utils.data import DataLoader

train_loader = DataLoader(                    # this basically gives us the data in batches to train our model
    train_dataset, 
    batch_size = 64,
    shuffle = True                            # randomly selected images
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

images, labels = next(iter(train_loader))

print("Images:", images.shape)
print("Labels:", labels.shape)


## Lets build our Convolutional Model
import torch.nn as nn

class MNISTCnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1, 
            out_channels = 16, 
            kernel_size = 3
        )

        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3
        )

        self.pool = nn.MaxPool2d(
            stride = 2,
            kernel_size = 2
        )

        self.fc = nn.Linear(            # self.fc is basically our classifier at the end
            32 * 5 * 5,                 # after we are done with the conv2d layer, we have 32 channels (see our number of filters above)
            10                          # and 10 is the final output vector (0-9 digits) over which we want to find which one does the digit belong to
        )

    def forward(self, x):               # input = [64, 1, 28, 28]
        x = self.conv1(x)               # [64, 16, 26, 26]
        x = torch.relu(x)               
        x = self.pool(x)                # [64, 16, 13, 13]

        x = self.conv2(x)               # [64, 32, 11, 11]
        x = torch.relu(x)               
        x = self.pool(x)                # [64, 32, 5, 5]

        x = x.flatten(1)                # converting our final matrix to a vector = [64, 800]
                                        # flatten everything starting from dimension 1
                                        # we need to keep each image, and flatten the rest of its dimensions 

        x = self.fc(x)                  # [64, 10]

        return x                        # returns logits and not raw probabilites, CrossEntropyLoss will handle them during training

model = MNISTCnn()

print(model)


## define loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.001
)

for epoch in range(50):                     # one epoch means we have gone through the entire 60k image training set once

    for images, labels in train_loader:     # training batch-by-batch

        # forward pass
        output = model(images)

        # calculate loss
        loss = loss_fn(output, labels)

        # clear old gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    print(
        f"Epoch: {epoch}"
        f"Loss:, {loss.item():.4f}"
    )

## Testing the model performance
model.eval()                            # switches our neural network into evaluation(inference) mode
                                        # doing this is important because certain layers behave differently during training and evaluation
                                        # prevent Dropout from happening (no random neurons need to be switched off during inference)
                                        # BatchNorm to use statistics leanred during training and not form the current batch

correct = 0
total = 0

with torch.no_grad():                          # we are not training anymore so don't calculate gradients, saves us memory, is efficient
    for images, labels in test_loader:

        output = model(images)

        predictions = output.argmax(dim=1)

        total += labels.size(0)                # adds the number of examples count = 64 into the running total
        correct += ((predictions == labels)
                    .sum()
                    .item())                   # assign 1 to correct predictions in this batch, sum all of them and retrieve

accuracy = correct/total

print(
    f"accuracy: {accuracy:.4f}"
)