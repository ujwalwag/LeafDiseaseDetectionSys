import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DATA_PATH = 'PlantVillage_Organized_Processed_Dataset/train'
VAL_DATA_PATH = 'PlantVillage_Organized_Processed_Dataset/test'
MODEL_SAVE_DIR = './trained_models'
BEST_MODEL_NAME = 'best_custom_vit_model.pth'
RESULTS_DIR = './results'

EFFECTIVE_CLASS_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]
NUM_CLASSES = len(EFFECTIVE_CLASS_LABELS)

LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class ViTDatasetTransform:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

print("Loading datasets...")
if not os.path.isdir(TRAIN_DATA_PATH):
    print(f"Error: Training data path '{TRAIN_DATA_PATH}' not found.")
    exit()
if not os.path.isdir(VAL_DATA_PATH):
    print(f"Error: Validation data path '{VAL_DATA_PATH}' not found.")
    exit()

try:
    custom_transform = ViTDatasetTransform(feature_extractor)

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DATA_PATH,
        transform=custom_transform
    )
    val_dataset = datasets.ImageFolder(
        root=VAL_DATA_PATH,
        transform=custom_transform
    )

    if train_dataset.classes != EFFECTIVE_CLASS_LABELS:
        print("Warning: Training dataset classes do not perfectly match EFFECTIVE_CLASS_LABELS.")
        print("Dataset classes:", train_dataset.classes)
        print("Expected classes:", EFFECTIVE_CLASS_LABELS)
    
    train_dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(EFFECTIVE_CLASS_LABELS)}
    val_dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(EFFECTIVE_CLASS_LABELS)}

    # Set num_workers to 0 to avoid multiprocessing issues on Windows
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {NUM_CLASSES}")

except Exception as e:
    print(f"Error loading datasets: {e}")
    exit()

print("Initializing ViT model...")
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

best_val_accuracy = 0.0
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
best_model_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train_predictions = 0
    total_train_predictions = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train_predictions += labels.size(0)
        correct_train_predictions += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_accuracy = correct_train_predictions / total_train_predictions
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    model.eval()
    val_running_loss = 0.0
    correct_val_predictions = 0
    total_val_predictions = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_val_predictions += labels.size(0)
            correct_val_predictions += (predicted == labels).sum().item()

            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_dataset)
    val_accuracy = correct_val_predictions / total_val_predictions
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with Validation Accuracy: {best_val_accuracy:.4f} to {best_model_path}")

print("\n--- Training Complete ---")
print(f"Best Validation Accuracy achieved: {best_val_accuracy:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_validation_metrics.png'))
plt.close()

print("\n--- Performing Final Evaluation on Test Dataset ---")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print(f"Loaded best model from {best_model_path} for final evaluation.")
else:
    print("Best model not found. Using the last trained model for final evaluation.")

all_final_preds = []
all_final_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(val_dataloader, desc="Final Evaluation"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).logits
        _, predicted = torch.max(outputs, 1)
        all_final_preds.extend(predicted.cpu().numpy())
        all_final_labels.extend(labels.cpu().numpy())

final_accuracy = accuracy_score(all_final_labels, all_final_preds)
print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")

print("\nFinal Classification Report:")
target_names = [train_dataset.classes[i] for i in sorted(train_dataset.class_to_idx.values())]
final_report = classification_report(all_final_labels, all_final_preds, target_names=target_names, digits=4)
print(final_report)

print("\nFinal Confusion Matrix:")
cm = confusion_matrix(all_final_labels, all_final_preds)
print(cm)

plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Final Evaluation)')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_vit.png'))
plt.close()

print("\nFinal Accuracy Per Class:")
class_accuracies = {}
for i, class_name in enumerate(target_names):
    true_positives = cm[i, i]
    total_instances_of_class = np.sum(cm[i, :])
    if total_instances_of_class > 0:
        class_accuracy = true_positives / total_instances_of_class
        class_accuracies[class_name] = class_accuracy
        print(f"  Class '{class_name}': {class_accuracy:.4f}")
    else:
        class_accuracies[class_name] = 0.0
        print(f"  Class '{class_name}': No instances found for this class in the test set.")

if class_accuracies:
    sorted_classes_acc = sorted(class_accuracies.items(), key=lambda item: item[1])
    plot_class_names = [item[0].replace('_', ' ').replace('___', ': ') for item in sorted_classes_acc]
    plot_accuracies = [item[1] for item in sorted_classes_acc]

    plt.figure(figsize=(12, 10))
    plt.barh(plot_class_names, plot_accuracies, color='lightgreen')
    plt.xlabel('Accuracy')
    plt.ylabel('Class Name')
    plt.title('Accuracy Per Class (Final Evaluation)')
    plt.xlim(0.0, 1.0)

    for index, value in enumerate(plot_accuracies):
        plt.text(value, index, f'{value:.4f}', va='center')

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_per_class_vit.png'))
    plt.close()

print("\nScript finished.")
