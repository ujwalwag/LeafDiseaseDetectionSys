# 🌿 Leaf Disease Detection System

This project is a web-based application that uses deep learning to identify and classify diseases in plant leaves from an uploaded image. The system leverages pre-trained Convolutional Neural Network (CNN) models and Vision Transformers (ViT) to provide fast and accurate predictions, making it a valuable tool for farmers, researchers, and gardening enthusiasts.

## ✨ Features

- **User-friendly Web Interface**: A simple and intuitive web page for uploading images and viewing results.
- **Real-time Prediction**: Get an instant diagnosis of a leaf's health with a confidence score.
- **Multiple Models**: The system supports various deep learning models, including ResNet50, VGG16, InceptionV3, and Vision Transformers, allowing for flexibility and performance comparison.
- **Detailed Insights**: Provides the predicted disease label, confidence score, and a description of the disease.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

You need to have Python 3.x and pip installed on your system.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/LEAFDISEASEDETECTIONSYS.git
    cd LEAFDISEASEDETECTIONSYS
    ```

2. Install the required libraries:

    This project uses libraries like PyTorch, Flask, and others. While a `requirements.txt` file is not shown, you would typically install them like this:

    ```bash
    pip install torch torchvision flask
    ```

    You may also need other libraries depending on the exact implementation.

### Running the Application

1. **Download Trained Models**: The pre-trained model weights (.pth files) are essential for making predictions. You may need to download these from a separate location or train them yourself using the provided training scripts. Place the `.pth` files in the `trained_models/` directory.

2. **Start the Backend Server**:

    ```bash
    python backend_app.py
    ```

    The server will start running, usually on `http://127.0.0.1:5000`.

3. **Open the Frontend**: Navigate to the `templates/` folder and open `index.html` in your web browser. The frontend will automatically connect to the running backend to perform predictions.

---

## 📂 Project Structure

The project is organized into the following key directories and files:

LEAFDISEASEDETECTIONSYS/
├── models/
│ ├── trained_models/
│ │ ├── best_resnet50_plant_disease_model.pth
│ │ └── ...
│ └── PlantVillage_Organized_Processed_Dataset
│ ├── test/
│ └── train/
├── templates/
│ └── index.html
├── scripts/
│ ├── app.py
│ ├── cpu_gpu_pytorch_test.py
│ ├── custom_vit.py
│ ├── desc_llm.py
│ ├── download.py
│ ├── huggingface_vit_tl.py
│ ├── InceptionV3_trainer.ipynb
│ ├── preprocess.py
│ ├── resnet_trainer.py
│ ├── vit_accuracy_per_class.py
│ └── README.md
├── backend_app.py
├── .gitattributes
└── README.md

- **models/**: Contains the model architectures and the processed dataset.
  - **PlantVillage_Organized_Processed_Dataset/**: The organized dataset used for training, split into `train/` and `test/` sets.
  - **trained_models/**: Stores the pre-trained model weights (`.pth` files).
  
- **templates/**: Holds the HTML and CSS for the frontend web interface.

- **scripts/**: A collection of Python scripts and notebooks for various development tasks.
  - `app.py`: Likely the main backend application for handling API requests and serving the model.
  - `cpu_gpu_pytorch_test.py`: A utility script to check if PyTorch is correctly configured to use a GPU, which is crucial for faster model training and inference.
  - `custom_vit.py`: Defines a custom Vision Transformer (ViT) model architecture, indicating experimentation with transformer-based models.
  - `desc_llm.py`: A script likely used for generating descriptive text or explanations for the detected diseases, possibly by leveraging a large language model (LLM).
  - `download.py`: A script for automatically downloading and preparing the raw dataset.
  - `huggingface_vit_tl.py`: A script that uses the Hugging Face Transformers library to fine-tune a pre-trained Vision Transformer model for the specific task of leaf disease detection.
  - `InceptionV3_trainer.ipynb`: A Jupyter Notebook for training the InceptionV3 model, which is useful for interactive development and visualization of the training process.
  - `preprocess.py`: Contains functions for pre-processing the raw image data, such as resizing, normalization, and data augmentation.
  - `resnet_trainer.py`: A script used to train the ResNet model on the dataset.
  - `vit_accuracy_per_class.py`: A script to evaluate the performance of a Vision Transformer model and calculate accuracy for each individual disease class.
  
- **backend_app.py**: The main Python server that handles image uploads and prediction logic.
  
- **.gitattributes**: A Git configuration file that specifies how certain file types should be treated, such as handling large binary files.

- **README.md**: The file you are currently reading.

---


