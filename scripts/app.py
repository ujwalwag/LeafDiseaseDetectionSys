import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, PhotoImage
import os


MODEL_SAVE_PATH = "best_resnet50_plant_disease_model_all_classes.pth"

IMG_HEIGHT, IMG_WIDTH = 224, 224


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


def get_resnet50_model(num_classes_model):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes_model)
    return model.to(device)


model = get_resnet50_model(NUM_CLASSES)

print(f"Attempting to load model from: {os.path.abspath(MODEL_SAVE_PATH)}") 

if os.path.exists(MODEL_SAVE_PATH):
    try:

        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval() 
        print(f"Model loaded successfully from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_SAVE_PATH}: {e}")
        print("Possible reasons: model file is corrupted, or it was saved with a different PyTorch version/architecture (e.g., number of output classes mismatch).")
        model = None 
else:
    print(f"Error: Model file '{MODEL_SAVE_PATH}' not found.")
    print("Please ensure the model file exists in the same directory as this script, or update MODEL_SAVE_PATH.")
    print("You need to run the training script (`resnet50_trainer_5_classes.py`) successfully first to create this file.")
    model = None 


preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def preprocess_image_for_prediction(image_path):
    """
    Loads an image from the given path, applies necessary transformations,
    and prepares it as a batch tensor for model inference.
    Returns the processed tensor and the original PIL image for display.
    """
    try:
        image = Image.open(image_path).convert("RGB") 
        input_tensor = preprocess_transform(image)

        input_batch = input_tensor.unsqueeze(0)
        return input_batch.to(device), image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


def display_image_in_tkinter(image_pil, predicted_class_label, confidence_score):
    """
    Updates the Tkinter GUI to display the image and prediction text.
    """
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
    """
    Handles image file selection, preprocessing, prediction, and GUI update.
    """
    if model is None:
        result_label.config(text="Model not loaded. Cannot predict.", fg="red")
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
            outputs = model(input_image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_prob, predicted_class_index = torch.max(probabilities, 1)

        predicted_class_label = EFFECTIVE_CLASS_LABELS_TRAINED[predicted_class_index.item()]
        confidence = predicted_prob.item() * 100

        display_image_in_tkinter(original_pil_image, predicted_class_label, confidence)

    except Exception as e:
        print(f"Error during prediction: {e}")
        result_label.config(text=f"Prediction Error: {e}", fg="red")
        confidence_label.config(text="", relief=tk.FLAT, bd=0)


app = tk.Tk()
app.title("Plant Disease Detector")
app.geometry("700x780") 
app.resizable(False, False)

app.configure(bg="#e0e0e0")
font_btn = ("Arial", 14, "bold")
font_label = ("Arial", 12)
font_result_main = ("Arial", 18, "bold")
font_confidence = ("Arial", 20, "bold") 

title_label = Label(app, text="PyTorch Plant Disease Detector",
                    font=("Arial", 20, "bold"), bg="#e0e0e0", fg="#333333")
title_label.pack(pady=20)

upload_button = Button(app, text="Upload Image for Detection", command=upload_and_predict,
                       font=font_btn, bg="#4CAF50", fg="white",
                       activebackground="#45a049", activeforeground="white",
                       relief=tk.RAISED, bd=3)
upload_button.pack(pady=20)

image_label = Label(app, bg="#ffffff", bd=2, relief=tk.SUNKEN)
image_label.pack(pady=10)

result_label = Label(app, text="Upload an image to see the prediction...",
                     font=font_result_main, bg="#e0e0e0", fg="#333333", wraplength=600)
result_label.pack(pady=10) 

confidence_label = Label(app, text="",
                         font=font_confidence, bg="#F0F8FF", fg="#006400", 
                         relief=tk.RIDGE, bd=2, padx=10, pady=5) 
confidence_label.pack(pady=10)  

app.mainloop()
