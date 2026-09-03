# Computer Vision From Scratch

This repository documents my journey of learning Computer Vision by implementing important concepts and architectures from first principles in PyTorch.

The goal is not simply to train models. It is to understand **why modern Computer Vision architectures work**, what problems they were designed to solve, and how those ideas translate into working code.

I am building each component incrementally, training models on real datasets, investigating their behavior, and using the results to understand the underlying concepts.

## Learning Philosophy

Rather than starting with pretrained models or high-level frameworks, I am working upward from the fundamentals:

**Convolution → CNNs → Optimization → Regularization → Batch Normalization → Deep Networks → Residual Connections → Modern Computer Vision**

For each major concept, I aim to:

1. Understand the mathematical and conceptual motivation.
2. Implement the idea myself in PyTorch.
3. Train it on a real dataset.
4. Inspect the model's behavior and failure modes.
5. Compare it with previous approaches.
6. Move toward increasingly sophisticated Computer Vision architectures.

The longer-term objective is to develop the foundation required to work on Computer Vision research.

## Current Progress

### 1. Image Convolution

Implemented the basic mechanics of image convolution from scratch to understand how local image operations produce feature maps.

### 2. CNN Fundamentals

Built simple convolutional neural networks and trained them on image classification tasks.

Topics explored include:

* Convolutional layers
* Kernels and feature maps
* Stride and padding
* Pooling
* Fully connected layers
* Forward propagation
* Loss functions
* Backpropagation
* Gradient-based optimization

### 3. MNIST

Built CNN models for handwritten digit classification and used the problem to understand the complete training loop.

### 4. CIFAR-10

Moved from grayscale MNIST images to RGB natural images.

Explored:

* RGB feature extraction
* Deeper CNN architectures
* Data augmentation
* Checkpointing
* Training and validation behavior

### 5. Data Augmentation

Investigated how transformations such as random cropping and horizontal flipping can improve generalization by exposing the model to different variations of the training images.

### 6. Batch Normalization

Implemented Batch Normalization in deeper CNNs and investigated its effect on optimization.

A major focus was understanding what happens to gradients as networks become deeper.

This led to an important observation: simply stacking more convolutional layers does not necessarily make a network easier to train.

### 7. Deep CNN on CIFAR-100

Built a deeper VGG-style CNN and trained it on CIFAR-100.

The experiment achieved approximately **54% test accuracy**, while also providing a useful opportunity to investigate gradient behavior throughout the network.

The experiment showed that Batch Normalization substantially improved gradient propagation compared with the corresponding plain deep network.

### 8. ResNet

The next major step is implementing **ResNet from scratch**.

The purpose is not just to reproduce the architecture, but to understand the problem that motivated residual connections.

The central question is:

> Why can adding more layers make a neural network harder to optimize, and how do residual connections address this?

The current implementation is based on the CIFAR-style ResNet architecture, using BasicBlocks and skip connections.

## Experiments

One of the recurring themes of this repository is comparing architectures rather than treating model performance as a single number.

For example:

| Model                | Dataset   | Main Concept                 |
| -------------------- | --------- | ---------------------------- |
| Simple CNN           | MNIST     | CNN fundamentals             |
| RGB CNN              | CIFAR-10  | Natural image classification |
| Augmented CNN        | CIFAR-10  | Data augmentation            |
| Deep CNN             | CIFAR-100 | Depth and optimization       |
| Deep CNN + BatchNorm | CIFAR-100 | Stable optimization          |
| ResNet               | CIFAR-100 | Residual learning            |

I also investigate quantities such as:

* Training loss
* Validation loss
* Test accuracy
* Gradient magnitudes
* Behavior of early and late layers
* Effects of architectural changes

This helps connect the theory of neural networks to what actually happens during training.

## Repository Structure

The scripts are organized roughly according to the progression of the learning process.

```text
Image_Detection_from_scratch/
│
├── 01_...                    # Image processing fundamentals
├── 02_...                    # Convolution fundamentals
├── ...
├── 03_...                    # Basic CNN
├── ...
├── 08_cifar100_deeper_CNN.py # Deep CNN on CIFAR-100
├── 09_cifar100_resnet_basic.py # ResNet implementation
│
├── data/                     # Local datasets
└── README.md
```

The exact structure will continue to evolve as the project progresses.

## Tools

* Python
* PyTorch
* Torchvision
* NumPy
* Git
* GitHub

## What Comes Next

The immediate goal is to complete and thoroughly understand the ResNet implementation.

After that, I plan to move progressively toward more modern Computer Vision concepts, including areas such as:

* More advanced CNN architectures
* Representation learning
* Object detection
* Image segmentation
* Vision Transformers
* Contrastive learning
* Vision-language models
* Computer Vision research papers

The ultimate goal is to move from **implementing known architectures** toward being able to **read, reproduce, investigate, and eventually contribute to Computer Vision research**.

## Why This Repository Exists

This repository is a record of the learning process.

The intention is to preserve not only successful models, but also failed experiments, debugging, unexpected results, and investigations into why a model behaves the way it does.

The progression matters:

**Understand the problem → understand the idea → implement it → experiment → investigate the result → move forward.**

This is the approach I am using to build a foundation for Computer Vision research.

```

I think this is much better than making the README sound like a conventional "ML project." It communicates that **the repository is documenting a deliberate progression toward research**, which fits what you're actually doing.

One thing I would change after we push this is the **Repository Structure** section. Once we inspect your actual files, we can replace the approximate structure with the exact sequence of scripts.
```
