from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

SPLIT_DIR = Path("data/splits_country")
OUTPUT_MODEL_DIR = Path("models/resnet50-country-split")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# Leicht erhoehtes Weight Decay, weil wir jetzt mehr Parameter trainierbar
# machen (mehr Kapazitaet = mehr Regularisierungsbedarf).
WEIGHT_DECAY = 2e-4
DROPOUT_RATE = 0.3
# Erhoeht von 1 auf 2: layer3 UND layer4 trainierbar. Der vorherige Lauf
# zeigte noch steigende Val-ROC-AUC bis zur letzten Epoche (0.82) - das
# deutet darauf hin, dass dem Modell noch Kapazitaet fehlte, um genug zu
# lernen, nicht dass es bereits ueberanpasste.
NUM_UNFROZEN_LAYERS = 2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DocumentDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


def build_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    resnet_blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    blocks_to_unfreeze = resnet_blocks[-NUM_UNFROZEN_LAYERS:]
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(num_features, 2),
    )

    return model


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds, target_names=["echt", "gefälscht"], zero_division=0)

    return avg_loss, accuracy, roc_auc, report


def main():
    print(f"Verwende Gerät: {DEVICE}")
    print(f"Modell: ResNet50, {NUM_UNFROZEN_LAYERS} von 4 Layer-Bloecken trainierbar, "
          f"Dropout={DROPOUT_RATE}, Weight Decay={WEIGHT_DECAY}\n")

    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_dataset = DocumentDataset(SPLIT_DIR / "train.csv", train_transform)
    val_dataset = DocumentDataset(SPLIT_DIR / "val.csv", eval_transform)
    test_dataset = DocumentDataset(SPLIT_DIR / "test.csv", eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training: {len(train_dataset)} | Validierung: {len(val_dataset)} | Test: {len(test_dataset)}")
    print("Hinweis: Die Test-Länder (lva, svk) kommen im Training NICHT vor.\n")

    model = build_model().to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainierbare Parameter: {num_trainable:,} von {num_total:,} gesamt\n")

    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Modellauswahl jetzt nach Validierungs-ROC-AUC statt Loss: Der letzte
    # Durchlauf zeigte, dass die Val-ROC-AUC bis zur letzten Epoche stetig
    # anstieg (0.52 -> 0.82), waehrend unser Loss-basiertes Kriterium
    # bereits bei Epoche 9 "stehen blieb" und ein schwaecheres Modell
    # gespeichert hat. Da ROC-AUC unsere eigentliche Zielmetrik ist,
    # richten wir die Modellauswahl direkt danach aus.
    best_val_auc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy, val_auc, _ = evaluate(model, val_loader)

        print(f"Epoche {epoch:2d}/{NUM_EPOCHS} - Trainings-Loss: {train_loss:.4f} - "
              f"Val-Loss: {val_loss:.4f} - Val-Genauigkeit: {val_accuracy:.4f} - Val-ROC-AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "best_model.pt")
            print(f"  -> Neues bestes Modell gespeichert (Val-ROC-AUC: {val_auc:.4f})")

    print("\n=== Lade bestes Modell für finale Test-Auswertung (UNBEKANNTE Länder!) ===")
    model.load_state_dict(torch.load(OUTPUT_MODEL_DIR / "best_model.pt", weights_only=True))
    _, test_accuracy, test_auc, test_report = evaluate(model, test_loader)

    print(f"\nTest-Genauigkeit: {test_accuracy:.4f}")
    print(f"Test-ROC-AUC: {test_auc:.4f}")
    print(f"\n{test_report}")


if __name__ == "__main__":
    main()