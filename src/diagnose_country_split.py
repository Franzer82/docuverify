from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn

SPLIT_DIR = Path("data/splits_country")
MODEL_PATH = Path("models/efficientnet-country-split/best_model.pt")

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model


def main():
    transform = build_transforms()
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            image = Image.open(row["filepath"]).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)
            output = model(image)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            pred = int(torch.argmax(output, dim=1).item())
            predictions.append(pred)
            probabilities.append(prob)

    test_df["prediction"] = predictions
    test_df["probability_fake"] = probabilities

    print("=== Verteilung der Vorhersagen (gesamt) ===")
    print(test_df["prediction"].value_counts().rename({0: "echt vorhergesagt", 1: "gefälscht vorhergesagt"}))

    print("\n=== Verteilung der vorhergesagten Wahrscheinlichkeiten ===")
    print(test_df["probability_fake"].describe())

    print("\n=== Genauigkeit PRO LAND ===")
    for country in sorted(test_df["country"].unique()):
        country_df = test_df[test_df["country"] == country]
        accuracy = (country_df["prediction"] == country_df["label"]).mean()
        print(f"{country}: {accuracy:.4f} ({len(country_df)} Bilder)")


if __name__ == "__main__":
    main()