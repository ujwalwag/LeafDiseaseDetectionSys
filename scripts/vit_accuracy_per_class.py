import matplotlib.pyplot as plt
import numpy as np

# Data provided by the user
accuracy_data = {
    'Apple___Apple_scab': 0.9921,
    'Apple___Black_rot': 0.9920,
    'Apple___Cedar_apple_rust': 1.0000,
    'Apple___healthy': 1.0000,
    'Blueberry___healthy': 1.0000,
    'Cherry_(including_sour)_Powdery_mildew': 1.0000,
    'Cherry_(including_sour)_healthy': 0.9942,
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': 0.9515,
    'Corn_(maize)Common_rust': 1.0000,
    'Corn_(maize)_Northern_Leaf_Blight': 0.9442,
    'Corn_(maize)_healthy': 1.0000,
    'Grape___Black_rot': 0.9450,
    'Grape__Esca(Black_Measles)': 1.0000,
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 1.0000,
    'Grape___healthy': 1.0000,
    'Orange__Haunglongbing(Citrus_greening)': 0.9900,
    'Peach___Bacterial_spot': 0.9650,
    'Peach___healthy': 1.0000,
    'Pepper,bell__Bacterial_spot': 0.9950,
    'Pepper,bell__healthy': 1.0000,
    'Potato___Early_blight': 1.0000,
    'Potato___Late_blight': 0.9900,
    'Potato___healthy': 0.7097,
    'Raspberry___healthy': 1.0000,
    'Soybean___healthy': 0.9950,
    'Squash___Powdery_mildew': 1.0000,
    'Strawberry___Leaf_scorch': 0.9950,
    'Strawberry___healthy': 0.9891,
    'Tomato___Bacterial_spot': 0.9650,
    'Tomato___Early_blight': 0.9800,
    'Tomato___Late_blight': 0.9700,
    'Tomato___Leaf_Mold': 1.0000,
    'Tomato___Septoria_leaf_spot': 0.9900,
    'Tomato___Spider_mites Two-spotted_spider_mite': 0.9950,
    'Tomato___Target_Spot': 0.9450,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 0.9950,
    'Tomato___Tomato_mosaic_virus': 1.0000,
    'Tomato___healthy': 0.9950
}

# Sort the data by accuracy for better visualization
sorted_classes = sorted(accuracy_data.items(), key=lambda item: item[1])
class_names = [item[0].replace('_', ' ').replace('___', ': ') for item in sorted_classes]
accuracies = [item[1] for item in sorted_classes]

# Create the horizontal bar chart
plt.figure(figsize=(12, 10)) # Adjust figure size for better readability of many classes
plt.barh(class_names, accuracies, color='skyblue')

# Add labels and title
plt.xlabel('Accuracy')
plt.ylabel('Class Name')
plt.title('Accuracy Per Class')
plt.xlim(0.0, 1.0) # Accuracy ranges from 0 to 1

# Add accuracy values on the bars for clarity
for index, value in enumerate(accuracies):
    plt.text(value, index, f'{value:.4f}', va='center')

plt.grid(axis='x', linestyle='--', alpha=0.7) # Add a grid for readability

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show() # Display the plot