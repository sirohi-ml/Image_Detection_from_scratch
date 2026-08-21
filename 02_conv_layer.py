import torch
import torch.nn as nn

## create a convolutional layer

conv = nn.Conv2d(                     # intializes weights and biases randomly
    in_channels = 1,                  # no. of channels in the input image, suggests a greyscale image
    out_channels = 1,                 # no. of channels in the output matrix 
                                      # or number of filters in short because that determines output channels
    kernel_size = 3                   # convolving filter size/dimensions
)

print(conv.weight)
print(conv.weight.shape)              # [out_channel, in_channel, height, width]
print(conv.bias)


## Add our image
image = torch.tensor([
    [1.,2.,3.,4.,5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.]
])

print(image.shape)

print(image)

image = image.unsqueeze(0).unsqueeze(0)  # this will add a dimension and then again a dimesion, because we want the image to be of shape
                                         # [out_channels, in_channels, height, width]

print(image.shape)

print(image)

output = conv(image)

print(output.shape)
print(output)


with torch.no_grad():
    conv.weight[:] = torch.tensor([
        [[[1., 0., -1.],
          [1., 0., -1.],
          [1., 0., -1.]]]
    ])

output = conv(image)
print(output)

print(conv.bias)
