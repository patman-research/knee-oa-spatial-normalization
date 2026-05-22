# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

#seed
SEED = 72
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# confiq
DATA_DIR = "/home/informatika/workspace/dataset/INASS_2_REV_3/4T_DETECT/dataset_resplit72"
BATCH_SIZE = 16
EPOCHS = 25
LR = 5e-5
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# transform
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# load dataset
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform)

g = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, generator=g)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, generator=g)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, generator=g)

# class weight
class_counts = np.array([2256, 1035, 1492, 720, 167])
class_weights = 1.0 / np.sqrt(class_counts)
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# model
def build_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        in_features = model.classif.in_features
        model.classif = nn.Linear(in_features, NUM_CLASSES - 1)

    else:
        raise ValueError("Not supported")

    return model.to(DEVICE)

# train
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = corn_loss(logits, labels, NUM_CLASSES)

        weights = class_weights[labels]
        loss = (loss * (1.0 + weights)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = corn_label_from_logits(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    return total_loss / len(loader), acc, kappa

# eval
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = corn_loss(logits, labels, NUM_CLASSES)

            weights = class_weights[labels]
            loss = (loss * (1.0 + weights)).mean()

            total_loss += loss.item()

            preds = corn_label_from_logits(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    return total_loss / len(loader), acc, kappa, spec_macro, all_preds, all_labels

# model list
model_list = ["resnet50",
              "densenet121",
              "efficientnet_b0",
              "convnext_tiny",
              "inception_resnet_v2"
             ]

# recap
results_summary = []

# loop
for model_name in model_list:
    print(f"\nTRAINING {model_name.upper()}")

    model = build_model(model_name)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3
    )

    MODEL_PATH = f"best_{model_name}_corn_seed_72_stratified.pth"

    best_kappa = 0
    patience = 7
    counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc, train_kappa = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_kappa, val_spec, _, _ = eval_epoch(model, val_loader)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")


        scheduler.step(val_kappa)

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model saved")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    print(f"\n===== TESTING {model_name.upper()} =====")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    _, _, _, spec_macro, preds, labels = eval_epoch(model, test_loader)

    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    acc = accuracy_score(labels, preds)
    mae = mean_absolute_error(labels, preds)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Kappa: {kappa:.4f}")
    print(f"Test MAE: {mae:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))

    results_summary.append({
        "Model": model_name,
        "Accuracy": acc,
        "Kappa": kappa,
        "MAE": mae
     })

# rekap table
print("\nRecap table")

results_summary = sorted(results_summary, key=lambda x: x["Kappa"], reverse=True)

print(f"{'Model':<25} {'Accuracy':<10} {'Kappa':<10} {'MAE':<10}")
print("-" * 70)

for res in results_summary:
    print(f"{res['Model']:<25} "
          f"{res['Accuracy']:<10.4f} "
          f"{res['Kappa']:<10.4f} "
          f"{res['MAE']:<10.4f} ")
   