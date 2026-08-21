import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels = 1,
    out_channels= 1,
    kernel_size=3
)

print("Initial weights:")
print(conv.weight)
print(conv.weight.shape)     # The print will mark a parameter "requires_grad" as True
                             # PyTorch is tracking these values because we want to calculate how they should change during training
print("Initial bias:")
print(conv.bias)


## adding our image 
image = torch.tensor([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.]
])

image = image.unsqueeze(0).unsqueeze(0)

output = conv(image)
print("Output:")
print(output)


## create a toy target tensor
target = torch.zeros_like(output)          # returns a tensor filled with scalar 0 of the same dimensions as the reference tensor

## calculate loss
loss_fn = nn.MSELoss()

loss = loss_fn(output, target)

print("Loss:", loss)


## Backpropagation, here we calculate the gradients or DL/DW
loss.backward()

print("Weight gradients:")
print(conv.weight.grad)

print("Bias gradient:")
print(conv.bias.grad)

## Updating the weights of the filter now
optimizer = torch.optim.SGD(                      # to use torch.optim we have to construct an optimizer object that holds the current state and will update the parameters based on the computed gradients
    conv.parameters(),
    lr = 0.001
)

optimizer.step()                                  # perform a single optimization step to update parameter

print("Updated weights:")
print(conv.weight)


## Training Loop

optimizer = torch.optim.SGD(
    conv.parameters(),
    lr = 0.0001
)

for epoch in range(100):

    output = conv(image)

    loss = loss_fn(output, target)

    optimizer.zero_grad()                # PyTorch accumulates gradients by default, We want each iteration to calculate fresh gradients
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

print("Final weights")
print(conv.weight)

print("Final bias")
print(conv.bias)