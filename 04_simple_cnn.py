import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels = 1,
            out_channels = 4,               # we are now creating 4 filters, for 4 feature maps
            kernel_size = 3
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(4,2)

    def forward(self, X):
        X = self.conv(X)
        X = torch.relu(X)
        X = self.pool(X)

        X = X.flatten(1)

        X = self.fc(X)                       

        return X                            # These are logits and not probabilities

model = SimpleCNN()                         # creating the model object

print(model)

## Defining Image
image = torch.tensor([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.]
])

image = image.unsqueeze(0).unsqueeze(0)
print("Input shape:", image.shape)
output = model(image)

print("Output shape:", output.shape)
print(output)


## Creating a Dataset

images = torch.tensor([
    # class 0
    [
        [5., 5., 1., 1., 1.],
        [5., 5., 1., 1., 1.],
        [5., 5., 1., 1., 1.],
        [5., 5., 1., 1., 1.],
        [5., 5., 1., 1., 1.]
    ],

    # class 1
    [
        [1., 1., 1., 5., 5.],
        [1., 1., 1., 5., 5.],
        [1., 1., 1., 5., 5.],
        [1., 1., 1., 5., 5.],
        [1., 1., 1., 5., 5.]
    ],

    [
            [1., 4., 1., 6., 5.],
            [1., 3., 1., 5., 4.],
            [1., 1., 1., 7., 5.],
            [1., 7., 1., 5., 3.],
            [1., 1., 9., 5., 8.]
        ]
])

labels = torch.tensor([0,1,1])

images = images.unsqueeze(1)

print(images.shape)
print(labels.shape)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.001
)

for epoch in range(500):

    output = model(images)
    loss = loss_fn(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        predictions = output.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        print(
            f"Epoch : {epoch}, " 
            f"Loss : {loss.item():.4f}, "
            f"Accuracy: {accuracy.item():.3f}"
            )


## Evaluating the model's generalization capabilities

test_image = torch.tensor([
    [
        [4., 4., 1., 1., 1.],
        [4., 4., 1., 1., 1.],
        [4., 4., 1., 1., 1.],
        [4., 4., 1., 1., 1.],
        [4., 4., 1., 1., 1.]
    ]
])

test_image = test_image.unsqueeze(0)

with torch.no_grad():
    prediction = model(test_image)

print("Test output: ", prediction)
print("Prediction class:", prediction.argmax(dim = 1).item())    # argmax() returns the position/index of the largest value
                                                                 # dim tells us we look across each row