import torch
print("torch imported successfully!")

import torchvision
print("torchvision imported successfully!")

import matplotlib
print("matplotlib imported successfully!")

import tqdm
print("tqdm imported successfully!")

from PIL import Image
print("Pillow (PIL) imported successfully!")

import cv2
print("opencv-python (cv2) imported successfully!")

import numpy as np
print("numpy imported successfully!")

if torch.cuda.is_available():
    print("PyTorch is GPU enabled.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is CPU enabled.")
    print("CUDA is not available, or PyTorch is not configured to use it.")

import torch
print("torch imported successfully!")

import torchvision
print("torchvision imported successfully!")

import matplotlib
print("matplotlib imported successfully!")

import tqdm
print("tqdm imported successfully!")

from PIL import Image
print("Pillow (PIL) imported successfully!")

import cv2
print("opencv-python (cv2) imported successfully!")

import numpy as np
print("numpy imported successfully!")
