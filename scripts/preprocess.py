import os
import shutil
import random
from tqdm import tqdm 
from PIL import Image 


SOURCE_DATA_ROOT = "PlantVillage-Dataset-Downloaded"


OUTPUT_DATA_ROOT = "PlantVillage_Organized_Processed_Dataset"


TRAIN_SPLIT_RATIO = 0.8

TARGET_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 224
TARGET_IMG_SIZE = (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT) 


CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def create_directory_structure(base_path, classes):

    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print(f"Created base directories: {train_path} and {test_path}")

    for cls in classes:
        os.makedirs(os.path.join(train_path, cls), exist_ok=True)
        os.makedirs(os.path.join(test_path, cls), exist_ok=True)


def process_and_save_image(image_path, target_size, save_path):

    try:
        with Image.open(image_path) as img:

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(save_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def distribute_dataset(source_root, output_root, classes, train_split_ratio, target_img_size):

    create_directory_structure(output_root, classes)

    print("\nStarting dataset distribution and initial image processing...")
    print(f"Images will be resized to {target_img_size[0]}x{target_img_size[1]} pixels and converted to RGB.")
    print("Please note: Full pixel value normalization (mean/std subtraction) is *not* done here.")
    print("That crucial step should still be performed dynamically during PyTorch data loading.")

    total_files_processed = 0
    for cls in tqdm(classes, desc="Processing classes"):
        class_source_path = os.path.join(source_root, cls)

        if not os.path.exists(class_source_path):
            print(f"Warning: Source path for class '{cls}' not found: {class_source_path}. Skipping.")
            continue

        images = [f for f in os.listdir(class_source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images) 

        num_train = int(len(images) * train_split_ratio)
        train_images = images[:num_train]
        test_images = images[num_train:]


        for img_name in train_images:
            src_path = os.path.join(class_source_path, img_name)
            dst_path = os.path.join(output_root, "train", cls, img_name)
            process_and_save_image(src_path, target_img_size, dst_path)
            total_files_processed += 1

        for img_name in test_images:
            src_path = os.path.join(class_source_path, img_name)
            dst_path = os.path.join(output_root, "test", cls, img_name)
            process_and_save_image(src_path, target_img_size, dst_path)
            total_files_processed += 1

    print(f"\nDataset distribution and initial processing complete! Total files processed: {total_files_processed}")
    print(f"Organized and resized dataset saved to: {os.path.abspath(output_root)}")
    print("Important Reminder: Final pixel normalization (mean/std subtraction) and data augmentation")
    print("should be applied by `torchvision.transforms` in your PyTorch training script during data loading.")

if __name__ == "__main__":

    if not os.path.exists(SOURCE_DATA_ROOT):
        print(f"Error: Source data root not found at '{SOURCE_DATA_ROOT}'")
        print("Please verify the path to your downloaded dataset (e.g., 'PlantVillage-Dataset-Downloaded/raw/color') and update SOURCE_DATA_ROOT in the script.")
    else:
        distribute_dataset(SOURCE_DATA_ROOT, OUTPUT_DATA_ROOT, CLASS_LABELS, TRAIN_SPLIT_RATIO, TARGET_IMG_SIZE)
