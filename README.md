# MNIST Digit-Predictor

## 📌 Overview
This project implements a digit classification model using `LeNet-5` architecture on the MNIST dataset. This CNN model is trained with 60,000
handwritten digits _0-9) grayscale image of 32x32 and successfully tested with the accuracy of 99.07% over 10000 handwritten digits. I used PyTorch framework to build the mode.
I used `torchvision` to load the datset, transforms to transform the data. Numpy for the mathematical computation and `torch.nn` for neural network.

At first, i loaded the MNIST dataset then after resized, transformed into tensor as tensor is as same as numpy but support GPU and finally Normalized the data. 
I introduced CNN model with 8 layers.

introduced first convolution layer. Convolution layer helps to convolv the input image data of size 32x32 using 6 kernels of size 5x5 each. added ReLU activation function to non-linearlity. And Kernel passed the 28x28x6 Tensor for pooling.
After maxpooling First Convolutional layer passed the output of size 14x14x6. 

In second Convolutional Layer, input of size (14x14x6) is convolved by 16 kernels and stride 1, introduced second ReLU activation function and then Max pooled with the 2x2 kernel with stride 2. Layer 2 passed the output of size 5x5x16.

Now can reached to the  Fully connected layer. It is also called dense layer which connects every inputs(here. 5x5x16) to every output neurons.
we use fully connected layer to reduce the number of parameter while training the model. 

I have 3 fully connected layers with different number of inputs and outputs. First fully connected layer get input of size 5x5x16. it is a linear layer pass the output of 120 units to the ReLU Function which help in capturing the non-linearity.

Second fc layer gets and inputs of 120 and pass output of size 84 and finally  last fully connected layer(softmax layer) passes the output of 10 difference digits classes (0-9).

For cost computation implemented the Cross entropy loss then back propagated and optimized with the Adam optimizer with is best among the optimizers.

Oh the main thing not to forget is Batch size. Batch size improves computational effieciency & parallelism. i defined 64 as batch size and ran 10 epochs for training.

Model is tested over 10000 test dataset, where model performed outstanding performance with 99.07% accuracy.

Lastly, Deployed model in FastAPI backend and for user interface used Nextjs. Frontend deployed in vercel.
User → Next.js Frontend → FastAPI Backend → ML Model

- Model architecture: LeNet-5 (Convolutional Neural Network)
- Dataset: MNIST(handwritten digit dataset)
- Input size : 32x32
- Framework: PyTorch

## 🏗 Model Architecture (LeNet-5)
1. Input layer: 32x32 grayscale image
2. Conv layer: 6 filter, 5x5 kernels, 1 stride, padding 0, Batch Normalization, ReLU activation function, Max Pooling(kernel size 2, stride 2, max pooling1)
3. Convolution layer ( 16 filters, 5x5 kernels, ReLU activation, Max Pooling2)
4. Fully Connected layer1: 120 units , ReLU
5. Fully Connect layer2: 84 units, ReLU
6. Fully Connect layer3: 10 classes, softmax function

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
