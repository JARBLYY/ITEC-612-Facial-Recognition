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
-  Will serve as the founfation for our facial recognition system.

Deliverable 2 Installation Instructions (The script runs best on Google Colab.)
- Download the preprocessed dataset zip file.
- Open Colab and import the preprocessed data zip file locally or through Google Drive.
- Install !pip install opencv-python dlib matplotlib tqdm face_recognition requests
  (Takes 2-5 minutes. Everything else (opencv, numpy, matplotlib, tqdm) is already on Colab.)
  - Download dlib's 68-point facial landmark predictor model if not already present
   url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
- Paste or import the entire pipeline script, and run it. Outputs (`preprocess_aligned.zip` and `face_embeddings.npz`) will auto-download when finished.

Features for Deliverable 2:
- Face Detection using dlib frontal face detector.
- Preprocessing (RGB format, aligned faces to consistent dimension),
- Face Extraction using bounding box coordinates.
- Eye landmark tilt correction.
- Face embedding generation.

Usage Examples: 
- Run the Full Face processing pipeline by running the Deliverable 2 ipynb file.
- The pipeline outputs face_embeddings.npz whre it can be used for face classfification, face verification, face identification, and embedding new images,
- Embeddings are stored in train, validation, and test splits, so they are ready to use.
  
License & Credits:
This project is for educational purposes. Group project for ITEC 612: Applications to Artificial Intelligence Course.
Developed by: Jarely Orozco (JARBLY), Alejandra Garcia (agarci6) , Ahmad Barakat (Abarakat), and Joe Castillo (joecastillo458). 
