import torch
from PIL import Image, UnidentifiedImageError
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import os
import numpy as np
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ViTDatasetTransform:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

    print("Loading pre-trained model and feature extractor...")
    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')

        base_model = ViTForImageClassification.from_pretrained(
            'wambugu1738/crop_leaf_diseases_vit',
            ignore_mismatched_sizes=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    except Exception as e:
        print(f"Error loading model or feature extractor: {e}")
        print("Please ensure you have internet access and the model/feature extractor paths are correct.")
        exit()

    train_dataset_path = 'PlantVillage_Organized_Processed_Dataset/train'
    test_dataset_path = 'PlantVillage_Organized_Processed_Dataset/test'

    if not os.path.isdir(train_dataset_path):
        print(f"Error: Training dataset path '{train_dataset_path}' does not exist or is not a directory.")
        print("Please ensure the directory structure is 'PlantVillage_Organized_Processed_Dataset/train/class_name/image.jpg'.")
        exit()
    if not os.path.isdir(test_dataset_path):
        print(f"Error: Test dataset path '{test_dataset_path}' does not exist or is not a directory.")
        print("Please ensure the directory structure is 'PlantVillage_Organized_Processed_Dataset/test/class_name/image.jpg'.")
        exit()

    print(f"Loading training dataset from: {train_dataset_path}")
    print(f"Loading test dataset from: {test_dataset_path}")

    try:
        custom_transform = ViTDatasetTransform(feature_extractor)

        train_dataset = datasets.ImageFolder(
            root=train_dataset_path,
            transform=custom_transform
        )
        print(f"Found {len(train_dataset)} images in the training dataset across {len(train_dataset.classes)} classes.")

        test_dataset = datasets.ImageFolder(
            root=test_dataset_path,
            transform=custom_transform
        )
        print(f"Found {len(test_dataset)} images in the test dataset across {len(test_dataset.classes)} classes.")

        if train_dataset.classes != test_dataset.classes:
            print("Warning: Training and test datasets have different class lists.")
            class_names = train_dataset.classes
            class_to_idx = train_dataset.class_to_idx
        else:
            class_names = train_dataset.classes
            class_to_idx = train_dataset.class_to_idx

        num_classes = len(class_names)
        print(f"Total {num_classes} classes identified: {class_names}")
        print("Class mapping (id2label):", {idx: label for label, idx in class_to_idx.items()})

        try:
            in_features = base_model.classifier.in_features
            base_model.classifier = nn.Linear(in_features, num_classes)
            print(f"Model classifier head replaced with a new Linear layer: {in_features} -> {num_classes} outputs.")
        except AttributeError:
            print("Could not find 'classifier.in_features'. Assuming output logits directly from base model.")

        base_model.to(device)

        batch_size = 32
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
        print(f"DataLoaders created with batch size: {batch_size}")

    except Exception as e:
        print(f"Error loading dataset or preparing model: {e}")
        print("Please ensure your datasets are structured correctly and images are valid.")
        exit()

    learning_rate = 2e-5
    num_epochs = 3

    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    print("\n--- Starting Fine-tuning ---")
    for epoch in range(num_epochs):
        base_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = base_model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_predictions / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch+1} finished. Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

    print("\n--- Fine-tuning Complete ---")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', linestyle='-')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    model_save_path = './fine_tuned_vit_model.pth'
    try:
        torch.save(base_model.state_dict(), model_save_path)
        print(f"\nFine-tuned model saved successfully to: {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n--- Starting Evaluation on Test Dataset ---")
    base_model.eval()
    all_preds = []
    all_labels = []
    evaluation_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = base_model(inputs).logits
            loss = criterion(outputs, labels)
            evaluation_loss += loss.item() * inputs.size(0)

            predictions = torch.argmax(outputs, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_evaluation_loss = evaluation_loss / len(test_dataset)
    print(f"Evaluation Complete. Test Loss: {avg_evaluation_loss:.4f}")

    if not all_labels:
        print("No predictions were made. Cannot evaluate further.")
    else:
        print("\n--- Final Model Evaluation Results (on Test Set) ---")

        true_labels = np.array(all_labels)
        predicted_labels = np.array(all_preds)

        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Overall Test Accuracy: {accuracy:.4f}")

        target_names = [class_names[i] for i in sorted(class_to_idx.values())]
        report = classification_report(true_labels, predicted_labels, target_names=target_names, digits=4)
        print("\nClassification Report:\n", report)

        cm = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion Matrix:")
        print(cm)
        print("Rows are True Labels, Columns are Predicted Labels.")
        print("To read: cm[i, j] is the number of instances of class i predicted as class j.")

        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        print("\nAccuracy Per Class:")
        for i, class_name in enumerate(target_names):
            true_positives = cm[i, i]
            total_instances_of_class = np.sum(cm[i, :])
            if total_instances_of_class > 0:
                class_accuracy = true_positives / total_instances_of_class
                print(f"  Class '{class_name}': {class_accuracy:.4f}")
            else:
                print(f"  Class '{class_name}': No instances found in dataset.")

    print("\nScript finished.")
