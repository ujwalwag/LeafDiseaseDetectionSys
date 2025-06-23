import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import uuid

ORGANIZED_DATA_ROOT = "PlantVillage_Organized_Processed_Dataset"

NUM_RANDOM_CLASSES = 5

IMG_HEIGHT, IMG_WIDTH = 224, 224

BATCH_SIZE = 32

GRADIENT_ACCUMULATION_STEPS = 4 

EPOCHS = 10

ALL_CLASS_LABELS = [
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

TEMP_DATA_DIR = "temp_5_class_dataset"
MODEL_SAVE_PATH = "best_resnet50_plant_disease_model.pth"

SCRIPT_RUN_ID = str(uuid.uuid4())[:8] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size=(IMG_HEIGHT, IMG_WIDTH), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

def get_resnet50_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def prepare_subset_data(source_root, temp_output_root, selected_classes):
    print(f"\nPreparing temporary dataset for {len(selected_classes)} selected classes...")
    print(f"Selected classes: {selected_classes}")

    temp_train_path = os.path.join(temp_output_root, "train")
    temp_val_path = os.path.join(temp_output_root, "val")

    if os.path.exists(temp_output_root):
        shutil.rmtree(temp_output_root)
    os.makedirs(temp_train_path, exist_ok=True)
    os.makedirs(temp_val_path, exist_ok=True)

    for phase in ['train', 'test']:
        source_phase_path = os.path.join(source_root, phase)
        temp_phase_path = os.path.join(temp_output_root, 'train' if phase == 'train' else 'val')

        for cls in tqdm(selected_classes, desc=f"Copying {phase} data"):
            source_class_path = os.path.join(source_phase_path, cls)
            temp_class_path = os.path.join(temp_phase_path, cls)
            os.makedirs(temp_class_path, exist_ok=True)

            if os.path.exists(source_class_path):
                for filename in os.listdir(source_class_path):
                    shutil.copy(os.path.join(source_class_path, filename), temp_class_path)
            else:
                print(f"Warning: Source class folder not found for {cls} in {source_phase_path}")
    print("Temporary dataset prepared.")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    all_val_preds = []
    all_val_labels = []
    best_val_accuracy = 0.0 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                all_val_preds_epoch = []
                all_val_labels_epoch = []

            running_loss = 0.0
            running_corrects = 0


            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=f"Phase {phase}")):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                 
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                        loss.backward()

                
                        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                            optimizer.step()
                            optimizer.zero_grad() 

                running_loss += loss.item() * inputs.size(0) * GRADIENT_ACCUMULATION_STEPS if phase == 'train' else loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'val':
                    all_val_preds_epoch.extend(preds.cpu().numpy())
                    all_val_labels_epoch.extend(labels.cpu().numpy())


            if phase == 'train' and (i + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else: 
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                all_val_preds = all_val_preds_epoch
                all_val_labels = all_val_labels_epoch


                if epoch_acc > best_val_accuracy:
                    best_val_accuracy = epoch_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"Saved best model to {MODEL_SAVE_PATH} with validation accuracy: {best_val_accuracy:.4f}")

        print()

    return model, history, all_val_preds, all_val_labels

def plot_graphs(history, num_epochs):
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_per_class_accuracy(true_labels, predicted_labels, class_names):
    report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    
    accuracies = {}
    for cls_name in class_names:
        if cls_name in report and 'recall' in report[cls_name]:
            accuracies[cls_name] = report[cls_name]['recall']
        else:
            accuracies[cls_name] = 0.0

    sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
    class_names_sorted = [item[0] for item in sorted_accuracies]
    accuracy_values_sorted = [item[1] for item in sorted_accuracies]

    plt.figure(figsize=(10, 6))
    plt.barh(class_names_sorted, accuracy_values_sorted, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(ALL_CLASS_LABELS) < NUM_RANDOM_CLASSES:
        raise ValueError(f"Cannot select {NUM_RANDOM_CLASSES} classes from only {len(ALL_CLASS_LABELS)} available.")
    selected_classes = random.sample(ALL_CLASS_LABELS, NUM_RANDOM_CLASSES)
    print(f"Randomly selected {NUM_RANDOM_CLASSES} classes: {selected_classes}")

    prepare_subset_data(ORGANIZED_DATA_ROOT, TEMP_DATA_DIR, selected_classes)

    image_datasets = {x: ImageFolder(os.path.join(TEMP_DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True if x == 'train' else False, num_workers=os.cpu_count())
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    effective_class_labels = image_datasets['train'].classes
    print(f"Effective classes loaded by ImageFolder: {effective_class_labels}")
    print(f"Number of classes for model: {len(effective_class_labels)}")

    model_ft = get_resnet50_model(len(effective_class_labels))

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)

    print("\nStarting model training...")
    model_ft, history, all_val_preds, all_val_labels = train_model(
        model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, num_epochs=EPOCHS
    )
    print("Training complete.")

    print("\nGenerating plots...")
    plot_graphs(history, EPOCHS)
    plot_confusion_matrix(all_val_labels, all_val_preds, effective_class_labels)
    plot_per_class_accuracy(all_val_labels, all_val_preds, effective_class_labels)

    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
        print(f"Cleaned up temporary data directory: {TEMP_DATA_DIR}")

    print("\nScript finished.")