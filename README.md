# TinyCNN CIFAR-10 Classification

## Overview
This project implements a **small Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**.  
CIFAR-10 consists of 60,000 32x32 color images from 10 classes, with 50,000 images for training and 10,000 images for testing.

The goal is to demonstrate:
- Building a CNN from scratch in PyTorch
- Training and evaluating a model on an image classification task
- Structuring code cleanly for readability and reproducibility

---

## Model Architecture
**TinyCNN** consists of:
- **Conv Layer 1**: 3 input channels -> 16 output channels, 3x3 kernel, ReLU activation
- **Max Pooling**: 2x2
- **Conv Layer 2**: 16 -> 32 channels, 3x3 kernel, ReLU activation
- **Max Pooling**: 2x2
- **Flatten Layer**
- **Fully Connected Stack**: 32 * 8 * 8 -> 128 -> 10 outputs (class scores) with ReLU between layers

---

## Requirements
- Python 3.8+
- PyTorch 2.x
- torchvision
- matplotlib(optional, for plotting, not used in current code)

Install dependencies using pip:
```bash
pip install torch torchvision matplotlib
```

---

## Usage
Run the training and evaluation script:
```bash
python tiny_cnn_cifar10.py
```
The script will:
1. Download CIFAR-10 dataset
2. Train TinyCNN for 5 epochs
3. Print per-epoch loss
4. Evaluate the model on the test set and print accuracy

Example output:
```yaml
Epoch 1, Loss: 1.4230544134174161
Epoch 2, Loss: 1.0786156521733765
...
Accuracy on CIFAR-10 test set: 69.09%
```

---

## Project Structure
```bash
CIFAR-10-Tiny-CNN/
├── tiny_cnn_cifar10.py       # Main training and evaluation script
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Notes
* This is a **baseline CNN** for CIFAR-10; performance is around **69% test accuracy** with 5 epochs.
* Potential improvements:
  - Batch normalization
  - Data augmentation
  - Learning rate scheduling
  - More convolutional layers
