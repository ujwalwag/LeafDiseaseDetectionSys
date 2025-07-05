from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import io

from transformers import ViTFeatureExtractor, ViTForImageClassification 


try:
    from scripts import desc_llm as llm_description_generator
except ImportError:
    print("Error: Could not import 'desc_llm.py' from 'scripts/' directory.")
    print("Please ensure 'desc_llm.py' is located in the 'scripts' folder relative to backend_app.py.")
    llm_description_generator = None

app = Flask(__name__)


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


loaded_models = {}
loaded_feature_extractors = {}
loaded_transforms = {}


def get_resnet50_model(num_classes_model):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes_model)
    return model

def get_inceptionv3_model(num_classes_model):
    model = models.inception_v3(pretrained=True, transform_input=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs_main = model.fc.in_features
    model.fc = nn.Linear(num_ftrs_main, num_classes_model)
    if model.AuxLogits is not None:
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes_model)
    return model

def get_vit_model(num_classes_model):
 
    model = ViTForImageClassification.from_pretrained(
        'wambugu1738/crop_leaf_diseases_vit',
        ignore_mismatched_sizes=True
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes_model)
    return model

def get_custom_vit_model(num_classes_model):
 
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes_model,
        ignore_mismatched_sizes=True
    )
    if not isinstance(model.classifier, nn.Linear):
         in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else model.config.hidden_size
         model.classifier = nn.Linear(in_features, num_classes_model)
    return model


def load_all_models():
    print("Loading all models on server startup...")
    for model_type, model_path in MODEL_PATHS_CONFIG.items():
        try:
            model_instance = None
            feature_extractor_instance = None
            transform_instance = None

            if model_type == "ResNet50":
                model_instance = get_resnet50_model(NUM_CLASSES)
                transform_instance = transforms.Compose([
                    transforms.Resize((IMG_HEIGHT_RESNET_VIT, IMG_WIDTH_RESNET_VIT)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif model_type == "InceptionV3":
                model_instance = get_inceptionv3_model(NUM_CLASSES)
                transform_instance = transforms.Compose([
                    transforms.Resize((IMG_HEIGHT_INCEPTION, IMG_WIDTH_INCEPTION)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif model_type == "ViT":
                model_instance = get_vit_model(NUM_CLASSES)

                feature_extractor_instance = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
            elif model_type == "Custom ViT":
                model_instance = get_custom_vit_model(NUM_CLASSES)
        
                feature_extractor_instance = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            if model_instance:
                if os.path.exists(model_path):
                    model_instance.load_state_dict(torch.load(model_path, map_location=device))
                    model_instance.eval()
                    model_instance.to(device)
                    loaded_models[model_type] = model_instance
                    loaded_feature_extractors[model_type] = feature_extractor_instance
                    loaded_transforms[model_type] = transform_instance
                    print(f"Successfully loaded {model_type} from {model_path}")
                else:
                    print(f"Warning: Model file '{model_path}' for {model_type} not found. Skipping.")
            else:
                print(f"Warning: Model instance for {model_type} could not be created. Skipping.")

        except Exception as e:
            print(f"Error loading {model_type} from {model_path}: {e}")
            print("Please ensure the model file exists and matches the architecture.")


with app.app_context():
    load_all_models()

@app.route('/')
def index():

    return render_template('index.html', model_options=list(MODEL_PATHS_CONFIG.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    model_type = request.form.get('model_type')
    if not model_type or model_type not in loaded_models:
        return jsonify({'error': 'Invalid or unloaded model type selected'}), 400

    current_model = loaded_models[model_type]
    current_feature_extractor = loaded_feature_extractors[model_type]
    current_preprocess_transform = loaded_transforms[model_type]

    try:
 
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

       
        input_batch = None
        if current_feature_extractor: 
            input_tensor = current_feature_extractor(images=image, return_tensors="pt")
            input_batch = input_tensor['pixel_values']
        elif current_preprocess_transform: 
            input_tensor = current_preprocess_transform(image)
            input_batch = input_tensor.unsqueeze(0)
        else:
            return jsonify({'error': 'Preprocessing method not found for selected model.'}), 500

        input_batch = input_batch.to(device)

        with torch.no_grad():
            model_output = current_model(input_batch)
            
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

      
        description = "LLM description generator not available. Check server setup."
        if llm_description_generator:
            description = llm_description_generator.generate_llm_description(predicted_class_label)
        
        return jsonify({
            'label': predicted_class_label,
            'confidence': f"{confidence:.2f}%",
            'description': description
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':

    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
