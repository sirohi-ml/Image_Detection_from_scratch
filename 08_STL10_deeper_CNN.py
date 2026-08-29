import torch 
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset

## Transforms
torch.manual_seed(42)

test_transform = v2.ToTensor()

train_transform = v2.Compose([
    v2.RandomResizedCrop(size = (32, 32)),     # 1.Randomly choose a region (20 x 30), 2.Resize that to 32 x32 image # re-sizing happens through interpolation i.e. if pixels did not exist in the original image, their values are estimated from nearby pixels (like average of 2 exisitng values)
    v2.RandomHorizontalFlip(0.5), 
    v2.ToTensor()
])

## Loading Datasets

train_full = datasets.CIFAR100(
    root = "./data",
    train=True, 
    download = True,
    transform=train_transform
)

validation_full = datasets.CIFAR100(
    root = "./data",
    train=True,
    download=True,
    transform=test_transform
)

test_full = datasets.CIFAR100(
    root = "./data",
    train=False,
    download=True,
    transform=test_transform
)

print("Training images:", len(train_full))
print("Validation images:", len(validation_full))
print("Test images:", len(test_full))

## Creating our train, val datasets
train_size = 45000
validation_size = 5000

g = torch.Generator().manual_seed(42)

indices = torch.randperm(                     # returns a random permutation of integers from 0 to (n-1)
    n = len(train_full),                      # parameters : n = upper bound
    generator=g                               # set the seed for reproducibility, 'g' is a way of settiung local seed
)

train_indices = indices[:train_size]
validation_indices = indices[train_size:]

train_data = Subset(                          # utility to create a subset of a dataset
    train_full,                               # the dataset to be sued
    train_indices                             # the indices
)

validation_data = Subset(
    validation_full,
    validation_indices
)

print("Final train length:", len(train_data))
print("Final val length:", len(validation_data))
print("Final test length:", len(test_full))

train_image, train_label = train_data[0]
print("Train image shape:", train_image.shape)
print("Train label:", train_label)

## Creating DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle=True
)

val_loader = DataLoader(
    validation_data,
    batch_size = 64, 
    shuffle = False
)

test_loader = DataLoader(
    test_full,
    batch_size = 64,
    shuffle=False
)

images, labels = next(iter(train_loader))
print("Train shape:", images.shape)
print("Label shape", labels.shape)


## Designing the CNN architecture (VGG 19 style)
class CIFAR100_deep_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1_2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(
            stride=2,
            kernel_size=2
        )

        self.conv2_1 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            stride=1,
            padding=1
        )

        self.bn2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(
            in_channels = 128, 
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(
            in_channels = 128,
            out_channels=256,
            kernel_size = 3,
            stride = 1,
            padding=1
        )

        self.bn3_1 = nn.BatchNorm2d(256)

        self.conv3_2 = nn.Conv2d(
            in_channels = 256,
            out_channels=256,
            kernel_size = 3,
            stride = 1,
            padding=1
        )

        self.bn3_2 = nn.BatchNorm2d(256)

        self.conv3_3 = nn.Conv2d(
            in_channels = 256,
            out_channels=256,
            kernel_size = 3,
            stride = 1,
            padding=1
        )

        self.bn3_3 = nn.BatchNorm2d(256)

        self.conv3_4 = nn.Conv2d(
            in_channels = 256,
            out_channels=256,
            kernel_size = 3,
            stride = 1,
            padding=1
        )

        self.bn3_4 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn4_1 = nn.BatchNorm2d(512)

        self.conv4_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn4_2 = nn.BatchNorm2d(512)

        self.conv4_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn4_3 = nn.BatchNorm2d(512)

        self.conv4_4 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn4_4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn5_1 = nn.BatchNorm2d(512)

        self.conv5_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn5_2 = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn5_3 = nn.BatchNorm2d(512)

        self.conv5_4 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn5_4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(
            512 * 1 * 1,
            4096
        )

        self.fc2 = nn.Linear(
            4096, 
            4096
        )

        self.fc3 = nn.Linear(
            4096,
            100
        )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = torch.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = torch.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = torch.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = torch.relu(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = torch.relu(x)
        x = self.conv3_4(x)
        x = self.bn3_4(x)
        x = torch.relu(x)
        x = self.pool(x) 

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = torch.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = torch.relu(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = torch.relu(x)
        x = self.conv4_4(x)
        x = self.bn4_4(x)
        x = torch.relu(x)
        x = self.pool(x) 

        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = torch.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = torch.relu(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = torch.relu(x)
        x = self.conv5_4(x)
        x = self.bn5_4(x)
        x = torch.relu(x)
        x = self.pool(x) 

        x = x.flatten(1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x

## Building the model
model = CIFAR100_deep_cnn()

criterion = nn.CrossEntropyLoss()

images, labels = next(iter(train_loader))
print("Input shape:", images.shape)

output = model(images)
print("Output shape:", output.shape)


## Training pipeline

model = CIFAR100_deep_cnn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

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
        # print("conv1_1:", model.conv1_1.weight.grad.abs().mean().item())
        # print("conv3_1:", model.conv3_1.weight.grad.abs().mean().item())
        # print("conv5_4:", model.conv5_4.weight.grad.abs().mean().item())
        # print("fc3:", model.fc3.weight.grad.abs().mean().item())
        optimizer.step()

        total_train_loss += loss.item()
        train_count += 1

    train_loss = total_train_loss/train_count

    model.eval()

    total_val_loss = 0
    val_count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            loss = criterion(output, labels)
            total_val_loss += loss.item()
            val_count += 1
        val_loss = total_val_loss/val_count
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                "best_cifar100_cnn_deep.pth"
            )

        print(
            f"Epoch: {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
                )

print("best epoch:", best_epoch)
print("Best val loss:", best_val_loss)


## Evaluation
total = 0
correct = 0
model.load_state_dict(
    torch.load(
        "best_cifar100_cnn_deep.pth",
        weights_only = True
        )
        )
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        predictions = output.argmax(1)
        correct += (predictions==labels).sum().item()
        total += labels.size(0)

accuracy = correct/total

print("Test accuracy (%):", accuracy*100)


