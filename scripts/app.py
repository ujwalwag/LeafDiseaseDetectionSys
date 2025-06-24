import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, PhotoImage, OptionMenu, StringVar
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification


IMG_HEIGHT_RESNET_VIT, IMG_WIDTH_RESNET_VIT = 224, 224
IMG_HEIGHT_INCEPTION, IMG_WIDTH_INCEPTION = 299, 299

EFFECTIVE_CLASS_LABELS_TRAINED = [
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
NUM_CLASSES = len(EFFECTIVE_CLASS_LABELS_TRAINED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


MODEL_PATHS_CONFIG = {
    "ResNet50": "best_resnet50_plant_disease_model_all_classes.pth",
    "InceptionV3": "best_inceptionv3_plant_disease_model.pth",
    "ViT": "best_fine_tuned_vit_model.pth"
}

current_model = None
current_preprocess_transform = None
current_feature_extractor = None

def get_resnet50_model(num_classes_model):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes_model)
    return model.to(device)

def get_inceptionv3_model(num_classes_model):
    model = models.inception_v3(weights=models.InceptionV3_Weights.IMAGENET1K_V1, transform_input=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs_main = model.fc.in_features
    model.fc = nn.Linear(num_ftrs_main, num_classes_model)
    if model.AuxLogits is not None:
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes_model)
    return model.to(device)

def get_vit_model(num_classes_model):
    model = ViTForImageClassification.from_pretrained(
        'wambugu1738/crop_leaf_diseases_vit',
        ignore_mismatched_sizes=True
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes_model)
    return model.to(device)

def load_and_setup_model(model_type, model_path):
    global current_model, current_preprocess_transform, current_feature_extractor

    current_model = None
    current_preprocess_transform = None
    current_feature_extractor = None
    
    result_label.config(text="Loading model...", fg="blue")
    confidence_label.config(text="", relief=tk.FLAT, bd=0)
    image_label.config(image=None)

    try:
        if model_type == "ResNet50":
            current_model = get_resnet50_model(NUM_CLASSES)
            current_preprocess_transform = transforms.Compose([
                transforms.Resize((IMG_HEIGHT_RESNET_VIT, IMG_WIDTH_RESNET_VIT)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_type == "InceptionV3":
            current_model = get_inceptionv3_model(NUM_CLASSES)
            current_preprocess_transform = transforms.Compose([
                transforms.Resize((IMG_HEIGHT_INCEPTION, IMG_WIDTH_INCEPTION)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_type == "ViT":
            current_model = get_vit_model(NUM_CLASSES)
            current_feature_extractor = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
            current_preprocess_transform = None
        else:
            result_label.config(text="Invalid model type selected.", fg="red")
            return

        print(f"Attempting to load {model_type} model from: {os.path.abspath(model_path)}")

        if os.path.exists(model_path):
            current_model.load_state_dict(torch.load(model_path, map_location=device))
            current_model.eval()
            result_label.config(text=f"{model_type} loaded successfully!", fg="green")
            print(f"Model loaded successfully from {model_path}")
        else:
            result_label.config(text=f"Error: Model file '{os.path.basename(model_path)}' not found.", fg="red")
            print(f"Error: Model file '{model_path}' not found.")
            current_model = None
            current_feature_extractor = None
            print("Please ensure the model file exists in the same directory as this script, or update MODEL_SAVE_PATH.")

    except Exception as e:
        result_label.config(text=f"Error loading {model_type}: {e}", fg="red")
        print(f"Error loading {model_type} from {model_path}: {e}")
        current_model = None
        current_feature_extractor = None
        print("Possible reasons: model file is corrupted, or it was saved with a different PyTorch version/architecture.")

def preprocess_image_for_prediction(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        if current_feature_extractor:
            input_tensor = current_feature_extractor(images=image, return_tensors="pt")
            input_batch = input_tensor['pixel_values']
        elif current_preprocess_transform:
            input_tensor = current_preprocess_transform(image)
            input_batch = input_tensor.unsqueeze(0)
        else:
            raise ValueError("No preprocessing method available. Model not loaded correctly.")

        return input_batch.to(device), image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def display_image_in_tkinter(image_pil, predicted_class_label, confidence_score):
    display_size = (300, 300)
    img_display = image_pil.resize(display_size, Image.Resampling.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(img_display)

    image_label.config(image=img_tk)
    image_label.image = img_tk 

    result_label.config(text=f"Predicted: {predicted_class_label}", fg="#333333", font=("Arial", 16, "bold"))
    
    confidence_label.config(text=f"Confidence: {confidence_score:.2f}%", 
                            fg="#006400", font=("Arial", 20, "bold"),
                            relief=tk.RIDGE, bd=2, padx=10, pady=5) 

def upload_and_predict():
    if current_model is None:
        result_label.config(text="No model loaded. Please select and load a model.", fg="red")
        confidence_label.config(text="", relief=tk.FLAT, bd=0)
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return 

    input_image_tensor, original_pil_image = preprocess_image_for_prediction(file_path)

    if input_image_tensor is None:
        result_label.config(text="Error processing image.", fg="red")
        confidence_label.config(text="", relief=tk.FLAT, bd=0)
        return

    try:
        with torch.no_grad(): 
            outputs = current_model(input_image_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_prob, predicted_class_index = torch.max(probabilities, 1)

        predicted_class_label = EFFECTIVE_CLASS_LABELS_TRAINED[predicted_class_index.item()]
        confidence = predicted_prob.item() * 100

        display_image_in_tkinter(original_pil_image, predicted_class_label, confidence)

    except Exception as e:
        print(f"Error during prediction: {e}")
        result_label.config(text=f"Prediction Error: {e}", fg="red")
        confidence_label.config(text="", relief=tk.FLAT, bd=0)

def switch_model(*args):
    selected_model_type = model_selection_var.get()
    model_path_to_load = MODEL_PATHS_CONFIG.get(selected_model_type, "")
    load_and_setup_model(selected_model_type, model_path_to_load)

app = tk.Tk()
app.title("Plant Disease Detector")
app.geometry("700x780") 
app.resizable(False, False)
app.configure(bg="#e0e0e0")

font_btn = ("Arial", 14, "bold")
font_label = ("Arial", 12)
font_result_main = ("Arial", 18, "bold")
font_confidence = ("Arial", 20, "bold") 
font_dropdown = ("Arial", 12)

title_label = Label(app, text="Leaf Disease Detection System",
                    font=("Arial", 20, "bold"), bg="#e0e0e0", fg="#333333")
title_label.pack(pady=20)

model_options = ["ResNet50", "InceptionV3", "ViT"]
model_selection_var = StringVar(app)
model_selection_var.set(model_options[0])

model_dropdown_label = Label(app, text="Select Model:", font=font_label, bg="#e0e0e0", fg="#333333")
model_dropdown_label.pack(pady=(0, 5))

model_dropdown = OptionMenu(app, model_selection_var, *model_options, command=switch_model)
model_dropdown.config(font=font_dropdown, bg="#cccccc", fg="#333333", activebackground="#a0a0a0")
model_dropdown.pack(pady=(0, 10))

upload_button = Button(app, text="Upload Image for Detection", command=upload_and_predict,
                        font=font_btn, bg="#4CAF50", fg="white",
                        activebackground="#45a049", activeforeground="white",
                        relief=tk.RAISED, bd=3)
upload_button.pack(pady=20)

image_label = Label(app, bg="#ffffff", bd=2, relief=tk.SUNKEN)
image_label.pack(pady=10)

result_label = Label(app, text="Select a model and upload an image...",
                      font=font_result_main, bg="#e0e0e0", fg="#333333", wraplength=600)
result_label.pack(pady=10) 

confidence_label = Label(app, text="",
                         font=font_confidence, bg="#F0F8FF", fg="#006400", 
                         relief=tk.RIDGE, bd=2, padx=10, pady=5) 
confidence_label.pack(pady=10) 

switch_model()

app.mainloop()
