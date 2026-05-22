# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from coral_pytorch.dataset import corn_label_from_logits

# SEED
SEED = 62
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CONFIG
DATA_DIR = "/home/informatika/workspace/dataset/INASS_2_REV_3/4T_DETECT/dataset_resplit62"
BATCH_SIZE = 16
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# TRANSFORM
test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# DATASET
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'set_test_seed_62_sigma_10'), transform=test_transform)

g = torch.Generator().manual_seed(SEED)

test_loader  = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=g
)

# MODEL BUILDER
def build_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=False)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, NUM_CLASSES - 1)

    elif model_name == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2', pretrained=False)
        in_features = model.classif.in_features
        model.classif = nn.Linear(in_features, NUM_CLASSES - 1)

    else:
        raise ValueError("Model not supported")

    return model.to(DEVICE)

# EVAL
def eval_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            preds = corn_label_from_logits(logits)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    mae = mean_absolute_error(all_labels, all_preds)

    return acc, kappa, mae, spec_macro, all_preds, all_labels

# MODEL LIST
model_list = [
    "convnext_tiny",
     "resnet50",
     "densenet121",
     "efficientnet_b0",
     "inception_resnet_v2"
]

# REKAP
results_summary = []

# LOOP TEST =====================
for model_name in model_list:
    print(f"\nTESTING {model_name.upper()} ")

    model = build_model(model_name)

    MODEL_PATH = f"best_{model_name}_corn_seed_62_stratified.pth"

    if not os.path.exists(MODEL_PATH):
        print(f" Model not found: {MODEL_PATH}")
        continue

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded")

    acc, kappa, mae, spec_macro, preds, labels = eval_model(model, test_loader)

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

print("\Rrecap_table")

results_summary = sorted(results_summary, key=lambda x: x["Kappa"], reverse=True)

print(f"{'Model':<25} {'Accuracy':<10} {'Kappa':<10} {'MAE':<10}")
print("-" * 70)

for res in results_summary:
    print(f"{res['Model']:<25} "
          f"{res['Accuracy']:<10.4f} "
          f"{res['Kappa']:<10.4f} "
          f"{res['MAE']:<10.4f} ")
