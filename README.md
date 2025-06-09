# LeafDiseaseDetectionSysLeaf Disease Detection System
This project aims to build a Leaf Disease Detection System using deep learning techniques. The system classifies healthy and diseased leaves to help agricultural professionals and enthusiasts identify plant health issues effectively.

🌱 Project Structure
kotlin
Copy
Edit
my_directory/
│
├── data/
│   ├── images/
│   │   ├── leaf1.jpg
│   │   └── leaf2.jpg
│   └── labels/
│       └── labels.csv
│
├── models/
│   └── cnn_model.h5
│
├── scripts/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── README.md
📂 Folders
data/

images/ — Contains the raw leaf images used for training and testing.

labels/ — Contains the label files (e.g., CSV) mapping images to disease classes.

models/

Stores trained deep learning models in .h5 format for future inference.

scripts/

train_model.py — Script for training the CNN model on the leaf dataset.

evaluate_model.py — Script for evaluating the trained model on the validation/test set.

results/

confusion_matrix.png — Visualization of classification performance.

classification_report.txt — Detailed metrics including accuracy, precision, recall, and F1 score.

🛠️ Requirements
Python 3.8+

TensorFlow/Keras

OpenCV

NumPy

Matplotlib

Sklearn

You can install the requirements using:

bash
Copy
Edit
pip install -r requirements.txt
(Make sure to create a requirements.txt listing the packages if needed.)

🚀 Usage
Data Preparation:
Place your leaf images in the data/images/ directory and the corresponding labels in data/labels/.

Model Training:
Run:

bash
Copy
Edit
python scripts/train_model.py
Model Evaluation:
After training, evaluate the model:

bash
Copy
Edit
python scripts/evaluate_model.py
Results:
Check the results/ folder for evaluation metrics and confusion matrix.

🎯 Goal
Achieve an accuracy of at least 80% on the validation set and enable reliable disease segmentation.

📚 References
PlantVillage Dataset

TensorFlow Documentation

OpenCV Documentation