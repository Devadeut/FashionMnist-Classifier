from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()
security = HTTPBasic()

# --- Basic Authentication Dependency ---
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "user"  # set your username
    correct_password = "pass"  # set your password
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="best_model.pth"):
    # Use ResNet18 with modifications for Fashion MNIST (1-channel input, 10 classes)
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Transformation Pipeline ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure image is grayscale
    transforms.Resize((32, 32)),                   # resize to model input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))           # normalize similar to training
])

# --- Class Names for Fashion MNIST ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
    try:
        # Read and preprocess the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # add batch dimension

        # Model prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

