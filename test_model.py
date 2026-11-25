from transformers import pipeline
from PIL import Image

MODEL_PATH = "./models/sdxl-detector"

print("Loading local AI model...")

detector = pipeline(
    "image-classification",
    model=MODEL_PATH,
    device=-1
)

print("AI MODEL LOADED SUCCESSFULLY!")

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")

results = detector(image)

print("\nAI DETECTION RESULT")

for item in results:
    print(item["label"], round(item["score"] * 100, 2), "%")