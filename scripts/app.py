import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, PhotoImage, OptionMenu, StringVar
from tkinter import ttk
import os
import threading

# Try importing the LLM description generator script with the new name
try:
    import desc_llm as llm_description_generator # Renamed import
except ImportError:
    print("Error: Could not import 'desc_llm.py'.")
    print("Please ensure 'desc_llm.py' is in the same directory as 'app.py'.")
    llm_description_generator = None

# Import the entire transformers module for explicit referencing
import transformers

# --- Configuration Constants ---
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
    "ViT": "fine_tuned_vit_model.pth",
    "Custom ViT": "best_custom_vit_model.pth"
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
    model = models.inception_v3(pretrained=True, transform_input=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs_main = model.fc.in_features
    model.fc = nn.Linear(num_ftrs_main, num_classes_model)
    if model.AuxLogits is not None:
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes_model)
    return model.to(device)

def get_vit_model(num_classes_model):
    model = transformers.ViTForImageClassification.from_pretrained(
        'wambugu1738/crop_leaf_diseases_vit',
        ignore_mismatched_sizes=True
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes_model)
    return model.to(device)

def get_custom_vit_model(num_classes_model):
    model = transformers.ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes_model,
        ignore_mismatched_sizes=True
    )
    if not isinstance(model.classifier, nn.Linear):
         in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else model.config.hidden_size
         model.classifier = nn.Linear(in_features, num_classes_model)
    return model.to(device)


def load_and_setup_model(model_type, model_path):
    global current_model, current_preprocess_transform, current_feature_extractor

    current_model = None
    current_preprocess_transform = None
    current_feature_extractor = None
    
    result_label.config(text="Loading model...")
    style.configure('Result.TLabel', foreground='blue')
    confidence_label.config(text="")
    disease_description_label.config(text="")
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
            current_feature_extractor = transformers.ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
            current_preprocess_transform = None
        elif model_type == "Custom ViT":
            current_model = get_custom_vit_model(NUM_CLASSES)
            current_feature_extractor = transformers.ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            current_preprocess_transform = None
        else:
            result_label.config(text="Invalid model type selected.")
            style.configure('Result.TLabel', foreground='red')
            return

        print(f"Attempting to load {model_type} model from: {os.path.abspath(model_path)}")

        if os.path.exists(model_path):
            current_model.load_state_dict(torch.load(model_path, map_location=device))
            current_model.eval()
            result_label.config(text=f"{model_type} loaded successfully!")
            style.configure('Result.TLabel', foreground='green')
            print(f"Model loaded successfully from {model_path}")
        else:
            result_label.config(text=f"Error: Model file '{os.path.basename(model_path)}' not found.")
            style.configure('Result.TLabel', foreground='red')
            print(f"Error: Model file '{model_path}' not found.")
            current_model = None
            current_feature_extractor = None
            print("Please ensure the model file exists in the same directory as this script, or update MODEL_PATHS_CONFIG.")

    except Exception as e:
        result_label.config(text=f"Error loading {model_type}: {e}")
        style.configure('Result.TLabel', foreground='red')
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

def update_description_label(description):
    disease_description_label.config(text=description)
    style.configure('Description.TLabel', foreground='#555555')

def display_image_in_tkinter(image_pil, predicted_class_label, confidence_score):
    display_size = (300, 300)
    img_display = image_pil.resize(display_size, Image.Resampling.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(img_display)

    image_label.config(image=img_tk)
    image_label.image = img_tk 

    result_label.config(text=f"Predicted: {predicted_class_label}")
    style.configure('Result.TLabel', foreground='#333333')
    
    confidence_label.config(text=f"Confidence: {confidence_score:.2f}%")
    style.configure('Confidence.TLabel', foreground='#006400')
    
    disease_description_label.config(text="Generating description...")
    style.configure('Description.TLabel', foreground='#888888')
    
    if llm_description_generator:
        threading.Thread(target=lambda: app.after(0, update_description_label, llm_description_generator.generate_llm_description(predicted_class_label))).start()
    else:
        update_description_label("LLM description generator not available. Check script setup.")


def upload_and_predict():
    if current_model is None:
        result_label.config(text="No model loaded. Please select and load a model.")
        style.configure('Result.TLabel', foreground='red')
        confidence_label.config(text="")
        disease_description_label.config(text="")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return 

    input_image_tensor, original_pil_image = preprocess_image_for_prediction(file_path)

    if input_image_tensor is None:
        result_label.config(text="Error processing image.")
        style.configure('Result.TLabel', foreground='red')
        confidence_label.config(text="")
        disease_description_label.config(text="")
        return

    try:
        with torch.no_grad(): 
            model_output = current_model(input_image_tensor)
            
            if hasattr(model_output, 'logits'):
                logits = model_output.logits
            elif isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_prob, predicted_class_index = torch.max(probabilities, 1)

        predicted_class_label = EFFECTIVE_CLASS_LABELS_TRAINED[predicted_class_index.item()]
        confidence = predicted_prob.item() * 100

        display_image_in_tkinter(original_pil_image, predicted_class_label, confidence)

    except Exception as e:
        print(f"Error during prediction: {e}")
        result_label.config(text=f"Prediction Error: {e}")
        style.configure('Result.TLabel', foreground='red')
        confidence_label.config(text="")
        disease_description_label.config(text="")

def switch_model(*args):
    selected_model_type = model_selection_var.get()
    model_path_to_load = MODEL_PATHS_CONFIG.get(selected_model_type, "")
    load_and_setup_model(selected_model_type, model_path_to_load)

app = tk.Tk()
app.title("Plant Disease Detector")
app.geometry("1400x1000") # Increased default size
app.resizable(True, True) # Made resizable

style = ttk.Style(app)
style.theme_use('clam')

style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', foreground='#333333', font=('Arial', 12))

style.configure('Title.TLabel', font=("Arial", 24, "bold"), foreground='#333333')
style.configure('ModelSelect.TLabel', font=('Arial', 14, 'bold'), foreground='#333333')
style.configure('Result.TLabel', font=("Arial", 18, "bold"), foreground='#333333', wraplength=1200) # Increased wraplength
style.configure('Confidence.TLabel', font=("Arial", 20, "bold"), background='#F0F8FF', relief='flat', borderwidth=0, padding=(10, 5))
style.configure('Description.TLabel', font=("Arial", 12, "italic"), foreground='#555555', wraplength=1200) # Increased wraplength

style.configure('TButton', font=('Arial', 12, 'bold'), background='#4CAF50', foreground='white', relief='flat')
style.map('TButton', background=[('active', '#45a049')])

style.configure('TMenubutton', font=('Arial', 12), background='#cccccc', foreground='#333333', relief='flat')
style.map('TMenubutton', background=[('active', '#a0a0a0')])


main_frame = ttk.Frame(app, padding="20 20 20 20")
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(main_frame, text="Leaf Disease Detection System", style='Title.TLabel', anchor="center")
title_label.pack(pady=(0, 20))

model_options = ["ResNet50", "InceptionV3", "ViT", "Custom ViT"]
model_selection_var = StringVar(app)
model_selection_var.set(model_options[0])

model_dropdown_label = ttk.Label(main_frame, text="Select Model:", style='ModelSelect.TLabel')
model_dropdown_label.pack(pady=(0, 5))

model_dropdown = ttk.OptionMenu(main_frame, model_selection_var, model_options[0], *model_options, command=switch_model)
model_dropdown.config(width=20)
model_dropdown.pack(pady=(0, 20))

upload_button = ttk.Button(main_frame, text="Upload Image for Detection", command=upload_and_predict,
                           style='TButton')
upload_button.pack(pady=20)

image_label = ttk.Label(main_frame, background="#ffffff", relief="solid", borderwidth=1)
image_label.pack(pady=10, ipadx=5, ipady=5)

result_label = ttk.Label(main_frame, text="Select a model and upload an image...",
                         style='Result.TLabel', anchor="center")
result_label.pack(pady=(10, 5)) 

confidence_label = ttk.Label(main_frame, text="",
                             style='Confidence.TLabel', relief="ridge", borderwidth=2)
confidence_label.pack(pady=(5, 10)) 

disease_description_label = ttk.Label(main_frame, text="",
                                      style='Description.TLabel', justify=tk.CENTER, anchor="center")
disease_description_label.pack(pady=(10, 0))

switch_model()

app.mainloop()
