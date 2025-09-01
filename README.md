# MNIST Digit-Predictor

## 📌 Overview
This project implements a digit classification model using the LeNet-5 architecture on the MNIST dataset. The Convolutional Neural Network (CNN) model is trained on 60,000 handwritten grayscale digits (0-9), each resized to 32x32 pixels, and achieves 99.07% accuracy on the 10,000-digit test set.

The model is built with PyTorch, using torchvision for dataset loading and transformations, and numpy for mathematical computations.

## Workflow

1. Load the MNIST dataset.

2. Resize images to 32x32 and transform them into tensors (GPU-compatible).

3. Normalize the data.

4. Define the LeNet-5 CNN architecture with 8 layers.

5. Train the model using Cross-Entropy Loss and Adam optimizer.

6. Test the model for accuracy.

7. Deploy via FastAPI backend and Next.js frontend.

## 🏗 Model Architecture (LeNet-5)
1. Input Layer: 32x32 grayscale image
2. Conv Layer 1:
    - 6 filters, 5x5 kernels, stride 1, padding 0
    - Batch Normalization
    - ReLU activation
    - Max Pooling (2x2) → Output size: 14x14x6
3. Conv Layer 2:
    - 16 filters, 5x5 kernels, stride 1
    - ReLU activation
    - Max Pooling (2x2) → Output size: 5x5x16

4. Fully Connected Layer 1: 120 units, ReLU
5. Fully Connected Layer 2: 84 units, ReLU
6. Fully Connected Layer 3: 10 classes, Softmax

Fully connected layers reduce the number of parameters while capturing non-linearity.

## 🚀 Training
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Batch Size: 64
- Epochs: 10
- Accuracy: 99.07% on test set

## 🌐 Deployment
The trained model is deployed using:

-**FastAPI** → Backend API
- **Next.js** → Frontend

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/ujjwalbasnyat/MNIST-digit-Predictor.git
cd MNIST

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
