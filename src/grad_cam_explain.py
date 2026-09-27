from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = Path("models/resnet50-country-split/best_model.pt")
SPLIT_DIR = Path("data/splits_country")
OUTPUT_DIR = Path("data/gradcam_examples")

IMAGE_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model():
    """Muss exakt dieselbe Architektur wie beim Training sein, damit die
    gespeicherten Gewichte korrekt hineinpassen - welche Bloecke beim
    Training eingefroren waren, spielt fuer das reine Laden keine Rolle."""
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


def generate_gradcam_overlay(model, cam, image_path: Path, transform):
    """Erzeugt ein Bild, das die Original-Aufnahme mit der Grad-CAM-
    Heatmap überlagert - rote/warme Bereiche zeigen, wo das Modell seine
    Entscheidung am stärksten festgemacht hat."""
    original_image = Image.open(image_path).convert("RGB")
    original_resized = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    rgb_image = np.array(original_resized, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    return overlay


def main():
    print("Lade trainiertes Modell ...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = build_eval_transform()

    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    # Mehrere Beispiele aus BEIDEN Test-Ländern, um zu prüfen, ob das
    # Modell wirklich inhaltsabhängig oder nur positionsabhängig schaut
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = []
    for country in test_df["country"].unique():
        country_df = test_df[test_df["country"] == country]
        real_rows = country_df[country_df["label"] == 0]
        fake_rows = country_df[country_df["label"] == 1]
        if len(real_rows) > 0:
            examples.append((f"real_{country}", real_rows.iloc[0]))
        if len(fake_rows) > 0:
            examples.append((f"fake_{country}", fake_rows.iloc[0]))

    for name, row in examples:
        image_path = Path(row["filepath"])
        overlay = generate_gradcam_overlay(model, cam, image_path, transform)

        output_path = OUTPUT_DIR / f"gradcam_{name}.png"
        Image.fromarray(overlay).save(output_path)
        print(f"Gespeichert: {output_path} (Land: {row['country']}, "
              f"tatsächliches Label: {'gefälscht' if row['label'] == 1 else 'echt'})")

    print("\nÖffne die gespeicherten Bilder, um die Heatmaps zu betrachten.")


if __name__ == "__main__":
    main()