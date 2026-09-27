import json
import random
from pathlib import Path

from PIL import Image

DATA_DIR = Path("data/SIDTD/SIDTD/SIDTD/templates")
IMAGES_DIR = DATA_DIR / "Images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"


def load_real_sample():
    """Lädt ein zufälliges echtes Dokument."""
    real_files = sorted((IMAGES_DIR / "reals").glob("*.jpg"))
    sample_path = real_files[0]
    return sample_path


def load_fake_sample():
    """Lädt ein zufälliges gefälschtes Dokument samt Annotation."""
    fake_files = sorted((IMAGES_DIR / "fakes").glob("*.jpg"))
    sample_path = fake_files[0]

    annotation_path = ANNOTATIONS_DIR / "fakes" / f"{sample_path.stem}.json"
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    return sample_path, annotation


if __name__ == "__main__":
    print("=== Echtes Dokument ===")
    real_path = load_real_sample()
    real_image = Image.open(real_path)
    print(f"Datei: {real_path.name}")
    print(f"Bildgröße: {real_image.size}")

    print("\n=== Gefälschtes Dokument ===")
    fake_path, fake_annotation = load_fake_sample()
    fake_image = Image.open(fake_path)
    print(f"Datei: {fake_path.name}")
    print(f"Bildgröße: {fake_image.size}")
    print(f"Fälschungstyp: {fake_annotation['ctype']}")
    print(f"Manipuliertes Feld: {fake_annotation['field']}")
    print(f"Original-Quellbild: {fake_annotation['src']}")

    # Beide Bilder zur visuellen Kontrolle anzeigen
    real_image.show()
    fake_image.show()