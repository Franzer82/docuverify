from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = Path("models/resnet50-country-split/best_model.pt")

IMAGE_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LABELS = {0: "echt", 1: "gefälscht"}


def build_model():
    """Muss exakt dieselbe Architektur wie beim Training sein (2 von 4
    ResNet-Bloecken trainierbar - siehe train_classifier.py), damit die
    gespeicherten Gewichte korrekt hineinpassen. Fuer reine Inferenz
    spielt es keine Rolle, welche Teile beim Training eingefroren waren."""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 2),
    )
    return model


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_pipeline():
    """Lädt Modell + Grad-CAM einmalig. Rückgabe als Dictionary, damit die
    Streamlit-App das Ergebnis über @st.cache_resource wiederverwenden
    kann, statt bei jeder Interaktion neu zu laden."""
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = build_eval_transform()

    return {"model": model, "cam": cam, "transform": transform}


def predict_and_explain(image: Image.Image, pipeline: dict) -> dict:
    """Führt Vorhersage UND Grad-CAM-Erklärung in einem Schritt aus.

    Gibt zurück: die vorhergesagte Klasse, die Wahrscheinlichkeit für
    'gefälscht', sowie ein Overlay-Bild (Original + Heatmap), das direkt
    in der App angezeigt werden kann."""
    model = pipeline["model"]
    cam = pipeline["cam"]
    transform = pipeline["transform"]

    original_rgb = image.convert("RGB")
    original_resized = original_rgb.resize((IMAGE_SIZE, IMAGE_SIZE))

    input_tensor = transform(original_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        predicted_class = int(torch.argmax(output, dim=1).item())
        fake_probability = probs[1].item()

    # Grad-CAM braucht Gradienten, kann also NICHT unter torch.no_grad()
    # berechnet werden - deshalb hier als separater Schritt danach.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_array = np.array(original_resized, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True)
    overlay_image = Image.fromarray(overlay)

    return {
        "label": LABELS[predicted_class],
        "label_id": predicted_class,
        "fake_probability": fake_probability,
        "overlay_image": overlay_image,
    }