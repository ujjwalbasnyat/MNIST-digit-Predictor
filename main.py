import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
import torchvision.transforms as transforms
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware

# Initial fastAPI app
app = FastAPI()

origin = ['https://mnist-digit-predictor.vercel.app']

# Allow frontend origin
app.add_middleware(
   CORSMiddleware,
   allow_origins = origin,
   allow_credentials = True,
   allow_methods = ['*'],
   allow_headers = ['*'],
)


num_classes = 10
# Define the LeNet-5 model
class LeNet5(nn.Module):
  def __init__(self, num_classes):
    super(LeNet5, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(1,6, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride =2)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(6,16, kernel_size = 5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc = nn.Linear(400, 120)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(120, 84)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(84, num_classes)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.relu(out)
    out = self.fc1(out)
    out = self.relu1(out)
    out = self.fc2(out)
    return out


# Model loading
model = LeNet5(num_classes)
model.load_state_dict(torch.load('lenet_mnist_state.pth', map_location=torch.device('cpu')))
model.eval()


# Preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()

    # Concert bytes data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
       return {"error": "Invalid image"}
    # Resize to 28x28
    img = cv2.resize(img, (32, 32))

    # Convert to tensor and normalize
    img = transform(img).unsqueeze(0) # shape [ 1,1, 28, 28]

    # Inference
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, predicted = torch.max(probs,1)

    return {"digit": int(predicted.item()),"confidence": float(conf.item())}