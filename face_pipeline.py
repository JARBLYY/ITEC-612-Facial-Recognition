"""
ITEC 612 - Facial Recognition Project
Deliverable 2: Face Processing Pipeline

This script implements the complete face-processing pipeline:
1. Face Detection - Using face_recognition library (dlib-based HOG detector)
2. Face Extraction - Crop detected faces from images (in-memory)
3. Face Alignment - Eye landmark detection and tilt correction
4. Preprocessing - Resize to 224x224, convert to RGB, normalize
5. Feature Engineering - Generate 128-d face embeddings
6. Visualization - Side-by-side pipeline progression for sample images

Authors: Jarely Orozco, Alejandra Garcia, Ahmad Barakat, Joe Castillo
"""

import os
import sys
import json
import numpy as np
import cv2
import face_recognition
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "preprocess_aligned")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "face_embeddings.npz")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "visualizations")
FAILURE_REPORT = os.path.join(os.path.dirname(__file__), "failure_report.json")

TARGET_SIZE = (224, 224)  # Consistent output dimension for all faces

# Name normalization map: maps inconsistent folder names to a canonical label.
# This fixes the naming inconsistencies across train/test/val splits
# (e.g., "RDJ" in train vs "Robert Downey Jr T" in test).
NAME_MAP = {
    "angelina jolie": "Angelina_Jolie",
    "beyonce": "Beyonce",
    "beyonce t": "Beyonce",
    "beyonce v": "Beyonce",
    "jennifer lawrence": "Jennifer_Lawrence",
    "megan fox": "Megan_Fox",
    "niel": "Neil_Patrick_Harris",
    "niel patrick harris t": "Neil_Patrick_Harris",
    "niel patrick harris v": "Neil_Patrick_Harris",
    "rdj": "Robert_Downey_Jr",
    "robert downey jr t": "Robert_Downey_Jr",
    "robert downey jr v": "Robert_Downey_Jr",
    "taylor": "Taylor_Swift",
    "taylor switft": "Taylor_Swift",  # fixing the typo too
    "taylor swift v": "Taylor_Swift",
    "tom hanks": "Tom_Hanks",
    "tom holland": "Tom_Holland",
    "tom holland t": "Tom_Holland",
    "tom holland v": "Tom_Holland",
    "will smith": "Will_Smith",
}


def normalize_label(folder_name):
    """
    Convert a raw folder name into a canonical label.
    Handles inconsistent naming across train/test/val splits.

    Args:
        folder_name (str): The raw folder name from the dataset

    Returns:
        str: Normalized label for the person
    """
    key = folder_name.strip().lower()
    if key in NAME_MAP:
        return NAME_MAP[key]
    # Fallback: replace spaces with underscores
    return folder_name.strip().replace(" ", "_")


def detect_split(path):
    """
    Determine which dataset split (train/val/test) a file belongs to
    based on its file path.

    Args:
        path (str): Full file path of an image

    Returns:
        str: One of 'train', 'val', or 'test'
    """
    path_lower = path.lower()
    if "train" in path_lower:
        return "train"
    elif "val" in path_lower:
        return "val"
    elif "test" in path_lower:
        return "test"
    return "unknown"


# ============================================================
# STEP 1: FACE DETECTION
# ============================================================

def detect_faces(image_rgb):
    """
    Detect faces in an RGB image using the face_recognition library
    (dlib HOG-based detector). Returns bounding box coordinates.

    Args:
        image_rgb (np.ndarray): RGB image as a NumPy array

    Returns:
        list: List of (top, right, bottom, left) bounding box tuples
    """
    face_locations = face_recognition.face_locations(image_rgb, model="hog")
    return face_locations


# ============================================================
# STEP 2: FACE EXTRACTION (IN-MEMORY)
# ============================================================

def extract_face(image_rgb, face_location):
    """
    Extract (crop) a face from an image using the bounding box coordinates.
    The face is kept in memory as a NumPy array.

    Args:
        image_rgb (np.ndarray): RGB image as a NumPy array
        face_location (tuple): (top, right, bottom, left) bounding box

    Returns:
        np.ndarray: Cropped face as a NumPy array
    """
    top, right, bottom, left = face_location
    face = image_rgb[top:bottom, left:right]
    return face


# ============================================================
# STEP 3: FACE ALIGNMENT
# ============================================================

def get_eye_centers(image_rgb, face_location):
    """
    Detect facial landmarks and compute eye center coordinates
    for alignment purposes.

    Args:
        image_rgb (np.ndarray): RGB image
        face_location (tuple): (top, right, bottom, left) bounding box

    Returns:
        tuple: (left_eye_center, right_eye_center) as (x, y) tuples,
               or (None, None) if landmarks not detected
    """
    landmarks = face_recognition.face_landmarks(image_rgb, [face_location])
    if not landmarks:
        return None, None

    lm = landmarks[0]
    if "left_eye" not in lm or "right_eye" not in lm:
        return None, None

    left_eye = np.mean(lm["left_eye"], axis=0).astype(int)
    right_eye = np.mean(lm["right_eye"], axis=0).astype(int)
    return tuple(left_eye), tuple(right_eye)


def align_face(image_rgb, face_location):
    """
    Perform face alignment using eye landmark detection and tilt correction.
    Rotates the image so that the eyes are horizontally aligned, then
    re-detects and crops the face.

    Args:
        image_rgb (np.ndarray): RGB image
        face_location (tuple): (top, right, bottom, left) bounding box

    Returns:
        np.ndarray or None: Aligned face crop, or None if alignment fails
    """
    left_eye, right_eye = get_eye_centers(image_rgb, face_location)

    if left_eye is None or right_eye is None:
        # Fallback: return extracted face without alignment
        return extract_face(image_rgb, face_location)

    # Calculate the angle between the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Compute rotation center (midpoint between eyes)
    eye_center = (float((left_eye[0] + right_eye[0]) // 2),
                  float((left_eye[1] + right_eye[1]) // 2))

    # Rotate the full image to correct tilt
    h, w = image_rgb.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, float(angle), 1.0)
    rotated = cv2.warpAffine(image_rgb, rotation_matrix, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # Re-detect face in the rotated image
    new_locations = face_recognition.face_locations(rotated, model="hog")
    if new_locations:
        top, right, bottom, left = new_locations[0]
        return rotated[top:bottom, left:right]
    else:
        # If re-detection fails, use original bounding box on rotated image
        top, right, bottom, left = face_location
        # Clamp to image boundaries
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)
        return rotated[top:bottom, left:right]


# ============================================================
# STEP 4: PREPROCESSING
# ============================================================

def preprocess_face(face_rgb):
    """
    Apply preprocessing steps to an aligned face image:
    - Resize to TARGET_SIZE (224x224)
    - Ensure RGB format
    - Normalize pixel values to [0, 1] range

    Args:
        face_rgb (np.ndarray): Aligned face image (RGB, uint8)

    Returns:
        np.ndarray: Preprocessed face image (224x224, RGB, float32 [0-1])
    """
    # Resize to consistent dimensions
    resized = cv2.resize(face_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Ensure RGB (should already be, but just in case)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def save_preprocessed_image(face_normalized, output_path):
    """
    Save a preprocessed (normalized) face image to disk as a PNG.
    Converts from float [0-1] back to uint8 [0-255] for saving.

    Args:
        face_normalized (np.ndarray): Preprocessed face (float32, [0-1])
        output_path (str): File path to save the image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_uint8 = (face_normalized * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img.save(output_path)


# ============================================================
# STEP 5: FEATURE ENGINEERING (EMBEDDING EXTRACTION)
# ============================================================

def extract_embedding(face_rgb):
    """
    Generate a 128-dimensional face embedding using face_recognition library.
    The face image is resized and padded so that face_recognition can
    detect and encode the face reliably.

    Args:
        face_rgb (np.ndarray): Face image as RGB NumPy array (uint8 or float32)

    Returns:
        np.ndarray or None: 128-d embedding vector, or None if encoding fails
    """
    # face_recognition expects uint8 images
    if face_rgb.dtype == np.float32 or face_rgb.dtype == np.float64:
        face_uint8 = (face_rgb * 255).astype(np.uint8)
    else:
        face_uint8 = face_rgb.copy()

    # Resize to a reasonable size for the encoder
    face_uint8 = cv2.resize(face_uint8, (200, 200))

    # Add padding around the face so face_recognition can detect it properly
    # (it needs some background around the face for landmark detection)
    padded = np.zeros((300, 300, 3), dtype=np.uint8)
    padded[50:250, 50:250] = face_uint8

    try:
        # Let face_recognition handle detection + encoding together
        encodings = face_recognition.face_encodings(padded)
        if encodings:
            return encodings[0]
    except Exception:
        pass

    # Fallback: try without padding
    try:
        encodings = face_recognition.face_encodings(face_uint8)
        if encodings:
            return encodings[0]
    except Exception:
        pass

    return None


# ============================================================
# STEP 6: VISUALIZATION
# ============================================================

def create_visualization(original, bbox, extracted, aligned, preprocessed,
                         label, image_name, save_path):
    """
    Create a side-by-side visualization showing pipeline progression:
    1. Original image with bounding box
    2. Extracted face
    3. Aligned face
    4. Preprocessed face

    Args:
        original (np.ndarray): Original RGB image
        bbox (tuple): (top, right, bottom, left) face bounding box
        extracted (np.ndarray): Cropped face
        aligned (np.ndarray): Aligned face
        preprocessed (np.ndarray): Preprocessed face (float [0-1])
        label (str): Person's name
        image_name (str): Source image filename
        save_path (str): Path to save the visualization PNG
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Pipeline: {label} - {image_name}", fontsize=14, fontweight='bold')

    # 1. Original with bounding box
    orig_with_box = original.copy()
    top, right, bottom, left = bbox
    cv2.rectangle(orig_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
    axes[0].imshow(orig_with_box)
    axes[0].set_title("1. Detected Face")
    axes[0].axis("off")

    # 2. Extracted face
    axes[1].imshow(extracted)
    axes[1].set_title("2. Extracted Face")
    axes[1].axis("off")

    # 3. Aligned face
    axes[2].imshow(aligned)
    axes[2].set_title("3. Aligned Face")
    axes[2].axis("off")

    # 4. Preprocessed face
    if preprocessed.dtype == np.float32 or preprocessed.dtype == np.float64:
        axes[3].imshow(np.clip(preprocessed, 0, 1))
    else:
        axes[3].imshow(preprocessed)
    axes[3].set_title("4. Preprocessed (224x224)")
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# MAIN PIPELINE
# ============================================================

def load_dataset(dataset_dir):
    """
    Load the dataset from the directory structure.
    Returns a list of dicts with image path, split, and label info.

    Expected structure:
        dataset/
            Facial_Recognition_Train/
                Person_Name/
                    image1.jpg
            Facial_Recognition_Test/
                ...
            Facial_Recognition_Val/
                ...

    Args:
        dataset_dir (str): Path to the dataset root directory

    Returns:
        list: List of dicts with keys 'path', 'split', 'label', 'raw_label'
    """
    entries = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for split_folder in sorted(os.listdir(dataset_dir)):
        split_path = os.path.join(dataset_dir, split_folder)
        if not os.path.isdir(split_path):
            continue

        split = detect_split(split_folder)
        if split == "unknown":
            continue

        for person_folder in sorted(os.listdir(split_path)):
            person_path = os.path.join(split_path, person_folder)
            if not os.path.isdir(person_path):
                continue

            label = normalize_label(person_folder)

            for img_file in sorted(os.listdir(person_path)):
                ext = os.path.splitext(img_file)[1].lower()
                if ext not in valid_extensions:
                    continue

                img_path = os.path.join(person_path, img_file)
                entries.append({
                    'path': img_path,
                    'split': split,
                    'label': label,
                    'raw_label': person_folder,
                    'filename': img_file
                })

    return entries


def run_pipeline():
    """
    Execute the complete face processing pipeline:
    1. Load all images from the dataset
    2. Detect, extract, align, and preprocess faces
    3. Generate embeddings
    4. Save preprocessed images to preprocess_aligned/
    5. Save embeddings to face_embeddings.npz
    6. Generate side-by-side visualizations
    7. Write failure report
    """
    print("=" * 60)
    print("ITEC 612 - Face Processing Pipeline")
    print("=" * 60)

    # Load dataset
    print("\n[1/6] Loading dataset...")
    entries = load_dataset(DATASET_DIR)
    print(f"  Found {len(entries)} images across splits:")
    split_counts = defaultdict(int)
    for e in entries:
        split_counts[e['split']] += 1
    for s, c in sorted(split_counts.items()):
        print(f"    {s}: {c} images")

    # Process all images
    print("\n[2/6] Processing images through pipeline...")
    results = {"train": {"embeddings": [], "labels": []},
               "val": {"embeddings": [], "labels": []},
               "test": {"embeddings": [], "labels": []}}

    failures = {"no_face_detected": [], "alignment_failed": [],
                "embedding_failed": [], "load_failed": []}

    viz_candidates = []  # Collect successful results for visualization
    processed_count = 0
    total = len(entries)

    for i, entry in enumerate(entries):
        img_path = entry['path']
        split = entry['split']
        label = entry['label']
        filename = entry['filename']

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i+1}/{total}: {label}/{filename}")

        # Load image
        try:
            image_rgb = face_recognition.load_image_file(img_path)
        except Exception as e:
            failures["load_failed"].append({
                "path": img_path, "error": str(e)
            })
            continue

        # Step 1: Detect faces
        face_locations = detect_faces(image_rgb)
        if not face_locations:
            failures["no_face_detected"].append({
                "path": img_path, "label": label, "split": split
            })
            continue

        # Use the largest detected face
        face_location = max(face_locations,
                           key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))

        # Step 2: Extract face (in-memory)
        extracted = extract_face(image_rgb, face_location)

        # Step 3: Align face
        aligned = align_face(image_rgb, face_location)
        if aligned is None or aligned.size == 0:
            failures["alignment_failed"].append({
                "path": img_path, "label": label, "split": split
            })
            # Use extracted face as fallback
            aligned = extracted

        # Step 4: Preprocess
        preprocessed = preprocess_face(aligned)

        # Save preprocessed image
        out_filename = f"{label}_{os.path.splitext(filename)[0]}.png"
        out_path = os.path.join(OUTPUT_DIR, split, label, out_filename)
        save_preprocessed_image(preprocessed, out_path)

        # Step 5: Extract embedding
        embedding = extract_embedding(aligned)
        if embedding is None:
            failures["embedding_failed"].append({
                "path": img_path, "label": label, "split": split
            })
        else:
            results[split]["embeddings"].append(embedding)
            results[split]["labels"].append(label)

        processed_count += 1

        # Collect candidates for visualization (from train split, one per person)
        viz_labels_seen = {v["label"] for v in viz_candidates}
        if split == "train" and len(viz_candidates) < 5 and label not in viz_labels_seen:
            viz_candidates.append({
                "original": image_rgb,
                "bbox": face_location,
                "extracted": extracted,
                "aligned": aligned,
                "preprocessed": preprocessed,
                "label": label,
                "filename": filename
            })

    print(f"\n  Processed: {processed_count}/{total} images successfully")

    # Save embeddings
    print("\n[3/6] Saving embeddings to face_embeddings.npz...")
    save_dict = {}
    for split_name in ["train", "val", "test"]:
        embeds = results[split_name]["embeddings"]
        labs = results[split_name]["labels"]
        if embeds:
            save_dict[f"{split_name}_embeddings"] = np.array(embeds)
            save_dict[f"{split_name}_labels"] = np.array(labs)
            print(f"    {split_name}: {len(embeds)} embeddings, dim={np.array(embeds).shape[1]}")
        else:
            save_dict[f"{split_name}_embeddings"] = np.array([])
            save_dict[f"{split_name}_labels"] = np.array([])
            print(f"    {split_name}: 0 embeddings (no successful extractions)")

    np.savez(EMBEDDINGS_FILE, **save_dict)
    print(f"  Saved to: {EMBEDDINGS_FILE}")

    # Generate visualizations
    print(f"\n[4/6] Generating {len(viz_candidates)} side-by-side visualizations...")
    os.makedirs(VIZ_DIR, exist_ok=True)
    for idx, viz in enumerate(viz_candidates):
        save_path = os.path.join(VIZ_DIR, f"pipeline_viz_{idx+1}_{viz['label']}.png")
        create_visualization(
            viz["original"], viz["bbox"], viz["extracted"],
            viz["aligned"], viz["preprocessed"],
            viz["label"], viz["filename"], save_path
        )
        print(f"    Saved: {os.path.basename(save_path)}")

    # Write failure report
    print("\n[5/6] Writing failure report...")
    failure_summary = {
        "total_images": total,
        "successfully_processed": processed_count,
        "no_face_detected": len(failures["no_face_detected"]),
        "alignment_failed": len(failures["alignment_failed"]),
        "embedding_failed": len(failures["embedding_failed"]),
        "load_failed": len(failures["load_failed"]),
        "details": failures
    }

    with open(FAILURE_REPORT, 'w') as f:
        json.dump(failure_summary, f, indent=2)
    print(f"  Saved to: {FAILURE_REPORT}")

    # Print summary
    print("\n[6/6] PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total images:          {total}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  No face detected:      {len(failures['no_face_detected'])}")
    print(f"  Alignment failures:    {len(failures['alignment_failed'])}")
    print(f"  Embedding failures:    {len(failures['embedding_failed'])}")
    print(f"  Load failures:         {len(failures['load_failed'])}")
    print(f"\n  Output directories:")
    print(f"    Preprocessed images: {OUTPUT_DIR}")
    print(f"    Embeddings file:     {EMBEDDINGS_FILE}")
    print(f"    Visualizations:      {VIZ_DIR}")
    print(f"    Failure report:      {FAILURE_REPORT}")
    print("=" * 60)
    print("Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()
