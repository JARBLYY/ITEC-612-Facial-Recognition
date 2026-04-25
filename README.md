# ITEC-612-Facial-Recognition

Introduction:

A facial recognition system that utilizes AI and Python libraries for dataset creation, image preprocessing, and performance evaluation. The project involves creating a structured dataset and implementing a preprocessing pipeline to prepare the uploaded imges for model training and analysis.

Installation Instructions:
- Git clone the repository.
- Navigate into the project folder.
- Optional but recommended: Create and activate a virtual environment. 
- Install dependencies. Install 'tqdm' depending on OS.

Features:
- Organized dataset (One folder per individual)
- Image preprocessing pipeline: Resizing 224x224, Color standardiziation (RGB), Pixel normalization (0-1).
- Modular and reusable Python Code.
- Supports integration with other AI Models.

Usage Examples: 
- Place images inside the dataset/ folder organized by train, val, and test. This project currently has uploaded images of celebrities. 
- When the script runs, the terminal will display progress for each folder and individual.
- After a message will show "✅ Preprocessing complete! Check the processed_data folder."

Deliverable 2 Installation Instructions:
- Import libraries
import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import face_recognition
import shutil
import random
import requests
from zipfile import ZipFile
-Install dependencies (!pip install opencv-python dlib matplotlib tqdm face_recognition requests.)

Features for Deliverable 2:
- Face Detection using dlib frontal face detector.
- Preprocessing (RGB format, aligned faces to consistent dimension),
- Face Extraction using bouding box coordinates.
- Eye landmark tilt correction.
- Face embedding generation.


Usage Examples: 
- Run the Full Face processing pipeline by running the Deliverable 2 py file.
- While running, the 
  
License & Credits:
This project is for educational purposes. Group project for ITEC 612: Applications to Artificial Intelligence Course.
Developed by: Jarely Orozco, Alejandra Garcia, Ahmad Barakat, and Joe Castillo. 
