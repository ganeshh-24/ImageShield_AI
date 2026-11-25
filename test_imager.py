import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/imager-1-tiny/model/model_imager-1-tiny.pth"

print("Loading Imager model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

# Load ResNet50 architecture
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(2048, 2)
)

model.load_state_dict(checkpoint["model"])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint["mean"],
        std=checkpoint["std"]
    )
])

print("Model loaded successfully!")

# Get image path
image_path = input("Enter image path: ")

try:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    real_score = probabilities[0].item()
    ai_score = probabilities[1].item()

    threshold = checkpoint["best_threshold"]

    print("\n--- IMAGER DETECTION RESULT ---")
    print("Real score:", round(real_score * 100, 2), "%")
    print("AI score:", round(ai_score * 100, 2), "%")
    print("Threshold:", threshold)

    if ai_score >= threshold:
        print("Prediction: AI Generated")
    else:
        print("Prediction: Real")

except FileNotFoundError:
    print("Error: Image file not found. Check the image path.")

except Exception as e:
    print("Error:", e)