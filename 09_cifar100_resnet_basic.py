import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset


## Setting up the GPU
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("Using:", device)


## Setting up the transforms
train_transforms = v2.Compose([                         # Compose simply means apply these operations one after another, in this exact order
    v2.RandomCrop(size = (32,32), padding = 4),         # adds a padding of 4 around the image and randomly chooses a 32 x 32 region
    v2.RandomHorizontalFlip(0.5),
    v2.ToImage(),                                       # This converts the input into a torchvision image that the newer pipeline expects
    v2.ToDtype(torch.float32, scale = True),            # images typically arrive as uint8, where pixel values are 0 - 255, we first convert them to float32 and scale them from 0 - 1
    v2.Normalize(                                       # normalize the pixels of each channel separately with the below mean and std (x-mean)/std for R,G,B
        mean=(0.5071, 0.4867, 0.4408),                  # these specific numbers are commonly used CIFAR-100 channel statistics
        std=(0.2675, 0.2562, 0.2761)
    )
])

test_transforms = v2.Compose([                          # No augmentations here, just conversion to torchvision image and datatype with scaling of pixel values
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2562, 0.2761)
    )
])


## Loading Datasets
train_full = datasets.CIFAR100(
    root = "./data",
    train=True,
    download=True,
    transform=train_transforms
)

val_full = datasets.CIFAR100(
    root="./data",
    train=True,
    download=False,
    transform=test_transforms
)

test_dataset = datasets.CIFAR100(
    root = "./data",
    train=False,
    download=True,
    transform=test_transforms
)


## Train-Val Split
g = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_full),
    generator=g
)

train_size = 45000

train_dataset = Subset(
    train_full,
    indices[:45000]
)

val_dataset = Subset(
    train_full,
    indices[45000:]
)


## DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    num_workers=2,                        # Data preparation will happen in parallel, each worker will load the image, transform it and give to GPU, so a batch will load onto GPU faster
    pin_memory=True,                      # A special type of CPU memory that allows CUDA to transfer data to GPU faster and more efficiently. We have to pair this with images.to(device, non_blocking=True)
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    shuffle=False
)


## Building a Basic Model
class BasicBlock(nn.Module):

    expansion = 1                                              # the final number of channels produced by the block is out_channels x expansion

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False                                          # when we pass the feature map into BN, the scale parameter acts as the "bias" and therefore it is not needed during the convolution block
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,                                           # we are not downsizing the image here
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(inplace=True)                       # we are using the module as opposed to the function torch.relu. Both are same mathematical operations but this is easier as it has been coded. Also inplace=True means PyTorch should modify the existing tensor

        ## Identity/shortcut
        ## We are making the shapes compatible

        if stride != 1 or in_channels != out_channels:          # we are asking "Does the input need to be transformed before adding to the output of the main path"
                                                                # suppose main path produces [64, 128, 16, 16] post a convolution having different number of filters and say stride = 2
            self.shortcut = nn.Sequential(                      # the input is [64, 64, 32, 32]. This input cannot be added to the above output as shapes don't match
                nn.Conv2d(
                    in_channels, 
                    out_channels,
                    kernel_size=1,                              # It is a 1x1 filter because the job of shortcut is not to extract spatial features of the image
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)                    # We are creating a small NN on the shortcut path: The end of this block will then allow the two shapes to match          
            )

        else:

            self.shortcut = nn.Identity()                       # otherwise no need for a modification, idnetity behaves like f(x) = x


    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out

## Building ResNet-18
class Resnet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100):   # 'block' tells ResNet which type of block to use, like the earlier BasicBlock we defined 

        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride = 1,
            padding=1,
            bias = False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        ## Residual Stages
        self.layer1 = self.make_layer(
            block,                               # define which block to use
            64,                                  # out channels
            num_blocks[0],                        
            stride = 1                           # spatial dimensions also remain the same
        )

        self.layer2 = self.make_layer(
            block,
            128,                                 # no of channels/filter is doubled
            num_blocks[1],
            stride = 2                           # spatial dimensions are halved (Imp pattern in ResNet, when spatial resolution decreases, feature channels increase)
        )

        self.layer3 = self.make_layer(
            block, 
            256,
            num_blocks[2],                        
            stride = 2 
        )

        self.layer4 = self.make_layer(
            block,
            512,
            num_blocks[3],
            stride = 2
        )

        ## Global Avg Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))             # for each of the 512 channels, we calculate the avg of all the pixel spatial values

        ## Classifier
        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes
        )

    def make_layer(
            self,
            block,
            out_channels,
            num_blocks,
            stride
    ):
        strides = [stride] + [1] * (num_blocks - 1)    # example [2] + [1] * (3 - 1) = [2] + [1]*(2) = [2] + [1,1] = [2,1,1], so the three blocks get stride 2,1,1 respectively

        layers = []                                    # create an empty python list

        for stride in strides:                         # loop over 2,1,1

            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride
                )
            )

            self.in_channels = (
                out_channels * block.expansion
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def ResNet18(num_classes=100):

    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes
    )


model = ResNet18(num_classes=100).to(device)

print(model)

images, labels = next(iter(train_loader))

images = images.to(device)

with torch.no_grad():
    output = model(images)

print("Input :", images.shape)
print("Output:", output.shape)

## Training
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(                               # this class Implements stochastic gradient descent 
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

