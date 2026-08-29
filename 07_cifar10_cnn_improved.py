import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

test_transform = transforms.ToTensor()          # for test dataset we do not need to perform any random augmentations

train_transform = transforms.Compose([          # "transforms.compose" composes several transforms together
    transforms.RandomCrop(32, padding=4),       # crop the image at a random location, original image = [32 x 32], add 4 pixels of padding = [40 x 40], crop back randomly from any position = [32 x 32], transformation applied at every image
    transforms.RandomHorizontalFlip(),          # horizontally flip the given image randomly with a given probability, default prob = 0.5
    transforms.ToTensor()                       # also we are performing "on the fly augmentation" i.e. modifying existing images and not add new ones
])

train_dataset_full = datasets.CIFAR10(          # applying crop and horizontal flip transformations
    root = "./data",
    train = True, 
    download=True,
    transform = train_transform
)

validation_dataset_full = datasets.CIFAR10(     # applying no transformations except conversion to Tensor
    root = "./data",
    train=True,                                 # we will be using True here because we want to split the validation data from the train itself. 
    download=True,
    transform=test_transform                   
)

test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True, 
    transform=test_transform
)

## Train and Val Split

train_size = 45000
validation_size = 5000                          # absolute lenghts must sum upto the total

generator = torch.Generator().manual_seed(42)   # to fix for the randomness

indices = torch.randperm(                       # randperm(n) returns a 1d tensor containing random permutations of integers from 0 to n-1, essentially shuffling our data
    len(train_dataset_full),                    # what should be our n in the case
    generator = generator                       
)

train_indices = indices[:train_size]
validation_indices = indices[train_size:]

from torch.utils.data import Subset

train_dataset = Subset(
    train_dataset_full,                         # 50,000 training images with augmentation
    train_indices
)

validation_dataset = Subset(
    validation_dataset_full,                    # same 50,000 training images, but no augmentation
    validation_indices
)

print("Train dataset size:", len(train_dataset))               # dataset contains collection of images and not tensor hence cannot use .shape here
print("Validation dataset size:", len(validation_dataset))
print("Test dataset size:", len(test_dataset))

train_image, train_label = train_dataset[0]
validation_image, validation_label = validation_dataset[0]

print("Training image shape:", train_image.shape)              # we have now retrieved one image which is a tensor, hence .shape will work here      
print("Validation image shape:", validation_image.shape)

print("Training label:", train_label)
print("Validation label:", validation_label)

## Creating Data Loaders

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle = True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size = 64,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size = 64,
    shuffle = False
)

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


## Building the model

class CNN_improved(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 10,
            kernel_size = 3
        )

        self.conv2 = nn.Conv2d(
            in_channels = 10,
            out_channels=16,
            kernel_size = 5
        )

        self.pool = nn.MaxPool2d(
            stride=2,
            kernel_size=2
        )

        self.fc = nn.Linear(
            16 * 5 * 5, 
            10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = torch.relu(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x

## loading model and checking shapes to be sure

model = CNN_improved()

images, labels = next(iter(train_loader))

output = model(images)

print("Input shape:", images.shape)
print("Output shape:", output.shape)

## Establishing our baseline model without any dropout, regularization etc

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.001
)

for epoch in range(10):

    model.train()                                          # putting the model in train mode, important for things like Dropout and BatchNorm

    total_train_loss = 0
    total_count = 0

    for images, labels in train_loader:

        output = model(images)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()                     # loss.item() gives us the average loss of each batch
        total_count += 1

    train_loss = total_train_loss/total_count


    model.eval()

    total_val_loss = 0
    val_count = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            output = model(images)
            loss = criterion(output, labels)

            total_val_loss += loss.item()
            val_count += 1

    val_loss = total_val_loss/val_count

    print(
        f"Epoch: {epoch}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val loss: {val_loss:.4f}, "
    )



## Adding "Check-pointing"
## The idea is model >> train >> is this the best val_loss? >> If yes, save the model

model = CNN_improved()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.001
)

criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
best_epoch = 0

for epoch in range(10):

    model.train()

    total_train_loss = 0
    train_count = 0

    for images, labels in train_loader:

        output = model(images)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_count += 1

    train_loss = total_train_loss/train_count

    model.eval()

    total_val_loss = 0 
    val_count = 0

    with torch.no_grad():
    
        for images, labels in validation_loader:

            output = model(images)

            loss = criterion(output, labels)

            total_val_loss += loss.item()

            val_count += 1

        val_loss = total_val_loss/val_count

        if val_loss <= best_val_loss:                         # adding condition for checkpointing
            best_val_loss = val_loss 
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                "best_cifar10_model.pth"
            )

        print(
            f"EPoch: {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

print("Best epoch:", best_epoch)
print("Best_validation_loss:", best_val_loss)

## Best model loading and Test Evaluation

model.load_state_dict(
    torch.load(
        "best_cifar10_model.pth",
        weights_only = True
    )
)

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

print("Test accuracy (%):", accuracy*100)



## Creating a new model with regularization parameters

class CNN_improved(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels=10,
            kernel_size = 3
        )

        self.bn1 = nn.BatchNorm2d(               # Batch Norm looks at the activations and calculates their "mean and variance" across the batch or spatial dimension
            num_features=10                      # so if the input size is [64, 10, 30, 30] the batch norm is calculated for each channel across 64 x 30 x 30 at once i.e 10 means and 10 variances (because each channel represents a particular learned feature)
        )                                        # also the normalized value is scaled with "gamma" and shifted using "beta". These two are learnable parameters for the model

        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=16,
            kernel_size=5
        )

        self.bn2 = nn.BatchNorm2d(               # during training (model.train()), batch norm uses the current batch's statistics.
            num_features=16                      # during evaluation (model.eval()), it uses running statistics accumulated during training instead of calculating them from the test batch, running means the mean of all the batches and variance of all the batches
        )                                        # suppose we evaluate on 1 image first and then 3 image first. For the same image if we were to calculate stats using the eval test, the normalization would be different in both cases. Since, we want deterministic outcomes batch norm during evaluation will use running stats from the training pipeline

        self.pool = nn.MaxPool2d(
            stride=2,
            kernel_size=2
        )

        self.dropout = nn.Dropout(
            p = 0.3
        )

        self.fc = nn.Linear(
            16 * 5 * 5, 
            10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)                           # running mean = (1-m) * running mean old + m*current batch mean >>> exponentially weighted running estimate
        x = torch.relu(x)                         # where m is the BatchNorm momentum (typically 0.1)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x


model = CNN_improved()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.001,
    weight_decay = 1e-4                 # basically of a way of telling our model "Don't let the weights become too large"
)                                       # Total objective = Training Loss + lambda * sum(weights ^^ 2), helps with overfitting as too large weights tend to memorize the training data

criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
best_epoch = 0

for epoch in range(100):

    model.train()

    total_train_loss = 0
    train_count = 0

    for images, labels in train_loader:

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_count += 1

    train_loss = total_train_loss/train_count

    model.eval()

    total_val_loss=0
    val_count = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            output = model(images)

            loss = criterion(output, labels)

            total_val_loss += loss.item()

            val_count += 1

        val_loss = total_val_loss/val_count

        if val_loss < best_val_loss:                # checkpointing
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                "best_cifar10_model.pth"
                        )
            
        print(
            f"Epoch: {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

print("Best epoch:", best_epoch)
print("Best_validation_loss:", best_val_loss)

model.load_state_dict(
    torch.load(
        "best_cifar10_model.pth",
        weights_only = True
    )
)

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

print("Test accuracy (%):", accuracy*100)
