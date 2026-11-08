# Conditional GAN for MNIST Digit Generation

This project implements a **Conditional Generative Adversarial Network (cGAN)** using PyTorch and PyTorch Lightning to generate synthetic handwritten digit images from the MNIST dataset. The implementation focuses on generating handwritten images of the digit **3**. A generator learns to produce realistic digit images, while a discriminator learns to distinguish real MNIST images from generated images. The digit label is provided as a condition to both networks, allowing the generation process to be guided by a target class.

## Dataset

**MNIST Handwritten Digits Dataset**

The implementation uses the training portion of MNIST and filters the dataset to include only images belonging to digit **3**. The images are converted to tensors and normalized to the range [-1, 1].

The dataset is downloaded automatically through `torchvision.datasets.MNIST`.

## Model Architecture

The project consists of two neural networks:

1. Generator: The generator takes a random noise vector and a class-label embedding as inputs.
2. Discriminator: The discriminator receives an MNIST image and its corresponding class-label embedding.

## Training and Monitoring

PyTorch Lightning manages the training process. The implementation includes:

1. Manual optimization of generator and discriminator.
2. Generator and discriminator loss tracking.
3. TensorBoard logging.
4. Model checkpointing based on generator loss.
5. Generation and visualization of images at the end of each training epoch.

