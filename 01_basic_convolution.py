import torch

image = torch.tensor([
    [1.,2.,3.,4.,5.],
    [6.,7.,8.,9.,10.],
    [11.,12.,13.,14.,15.],
    [16.,17.,18.,19.,20.],
    [21.,22.,23.,24.,25.]
])

print(image)

kernel = torch.tensor([                # Also called the "Filter" that will be applied 
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])                                     # These are also the parameters which can be learned using GD
                                       # just 9 params in this layer, so less, why?
                                       # Because of parameter sharing across the image and sparse connections

patch = image[0:3, 0:3]                # for the convolution operation we need the patch and filter to have same N_h, N_w, N_c

print("Patch:")
print(patch)

print("Kernel:")
print(kernel)

result = torch.sum(patch * kernel)      # Element wise multiplication and not matrix multiplication

print("Convolution result:")
print(result)

output = torch.zeros(3,3)               # we already know the output will be 3,3 matrix 
                                        # formula for dim = (n + 2p - f)/s + 1

for i in range(3):
    for j in range(3):
        patch = image[i:i+3, j:j+3]
        output[i, j] = torch.sum(patch * kernel)

print("Output:")
print(output)




## Now changing the image:
image = torch.tensor([
    [1., 1., 1., 5., 5.],
    [1., 1., 1., 5., 5.],
    [1., 1., 1., 5., 5.],
    [1., 1., 1., 5., 5.],
    [1., 1., 1., 5., 5.]
])

for i in range(3):
    for j in range(3):
        patch = image[i:i+3, j:j+3]
        output[i, j] = torch.sum(patch * kernel)

print("Output:")
print(output)

