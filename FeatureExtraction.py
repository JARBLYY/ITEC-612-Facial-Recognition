# Facial Recognition Feature Extraction Pipeline
# Deliverable 2: Detection -> Extraction -> Alignment -> Preprocessing -> Embeddings
# @Author Ahmad Barakat
 
import urllib.request
import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import face_recognition
import shutil
import random
 
 
# Download dlib's 68-point facial landmark predictor model if not already present
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
file_name = "shape_predictor_68_face_landmarks.dat.bz2"
 
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("Downloading landmark model...")
    urllib.request.urlretrieve(url, file_name)
 
    import bz2
    with bz2.BZ2File(file_name) as fr, open("shape_predictor_68_face_landmarks.dat", "wb") as fw:
        fw.write(fr.read())
    print("Model downloaded and extracted.")
 
# Reproducibility
random.seed(42)
np.random.seed(42)
 
# Settings
input_root = "processed_data"
output_dir = "preprocess_aligned"
embedding_file = "face_embeddings.npz"
IMG_SIZE = 224
NUM_SAMPLES = 5
 
print("Dataset Path:", input_root)
 
# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 
 
def detect_faces(img):
    """
    Run dlib's frontal face detector on an RGB image.
 
    Args:
        img: RGB image as a NumPy array.
 
    Returns:
        A list of dlib rectangles, one per detected face.
    """
    return detector(img)
 
 
def extract_face(img, rect, pad=20):
    """
    Crop a face region from an image based on a detected bounding rectangle,
    with optional padding around the face.
 
    Args:
        img: RGB image as a NumPy array.
        rect: dlib rectangle from detect_faces() representing face bounds.
        pad: Number of pixels to expand the bounding box on each side.
 
    Returns:
        Cropped face image as a NumPy array, or None if the crop is invalid.
    """
    x1 = max(rect.left() - pad, 0)
    y1 = max(rect.top() - pad, 0)
    x2 = min(rect.right() + pad, img.shape[1])
    y2 = min(rect.bottom() + pad, img.shape[0])
 
    face = img[y1:y2, x1:x2]
    return face if face.size != 0 else None
 
 
def align_face(face_img):
    """
    Align a face image so that the eyes are horizontally level. Uses dlib's
    68-point landmark predictor to find the eye centers, computes the rotation
    angle, and applies an affine warp to straighten the face.
 
    Args:
        face_img: Cropped RGB face image as a NumPy array.
 
    Returns:
        Aligned face image as a NumPy array, or None if alignment fails.
    """
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        landmarks = predictor(gray, rect)
 
        # Average the 6 landmark points for each eye to get eye centers
        left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
        right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
 
        # Calculate rotation angle from the line between the eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
 
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                       (left_eye[1] + right_eye[1]) / 2)
 
        # Rotate the image around the midpoint of the eyes to level them
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(
            face_img, M,
            (face_img.shape[1], face_img.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        return aligned
 
    except:
        return None
 
 
def preprocess(img):
    """
    Preprocess an aligned face image for the embedding model: resize to a fixed
    square size and normalize pixel values to the [0, 1] range.
 
    Args:
        img: Aligned RGB face image as a NumPy array.
 
    Returns:
        Preprocessed image as a float NumPy array with values in [0, 1],
        or None if input is None.
    """
    if img is None:
        return None
 
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img
 
 
def save_image(img, path):
    """
    Save a normalized image (values in [0, 1]) to disk as a standard image file.
    Creates parent directories as needed.
 
    Args:
        img: Image as a NumPy array with float values in [0, 1].
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, (img * 255).astype(np.uint8))
 
 
def get_embedding(img):
    """
    Generate a 128-dimensional face embedding using the face_recognition library.
 
    Args:
        img: Preprocessed face image as a NumPy array with values in [0, 1].
 
    Returns:
        128-dimensional embedding vector as a NumPy array, or None if no
        embedding could be generated.
    """
    img_uint8 = (img * 255).astype(np.uint8)
    enc = face_recognition.face_encodings(img_uint8)
    return enc[0] if len(enc) > 0 else None
 
 
# Gather all image paths from the dataset directory
image_paths = []
for root, dirs, files in os.walk(input_root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))
 
print("Total images:", len(image_paths))
 
# Storage for embeddings and labels, organized by split
embeddings = {
    "train_embeddings": [], "train_labels": [],
    "val_embeddings": [], "val_labels": [],
    "test_embeddings": [], "test_labels": []
}
 
train_paths, val_paths, test_paths = [], [], []
failed = []
 
# Main pipeline: process each image through the full chain
for img_path in tqdm(image_paths):
 
    img = cv2.imread(img_path)
    if img is None:
        failed.append((img_path, "read failed"))
        continue
 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    faces = detect_faces(img_rgb)
    if len(faces) == 0:
        failed.append((img_path, "no face"))
        continue
 
    face = extract_face(img_rgb, faces[0])
    if face is None:
        failed.append((img_path, "crop failed"))
        continue
 
    aligned = align_face(face)
    if aligned is None:
        failed.append((img_path, "alignment failed"))
        continue
 
    processed = preprocess(aligned)
    if processed is None:
        failed.append((img_path, "preprocess failed"))
        continue
 
    # Save the processed image
    rel_path = img_path.replace(input_root, "").lstrip("/")
    save_image(processed, os.path.join(output_dir, rel_path))
 
    # Generate the face embedding
    emb = get_embedding(processed)
    if emb is None:
        failed.append((img_path, "embedding failed"))
        continue
 
    label = os.path.basename(os.path.dirname(img_path))
 
    # Detect which split (train/val/test) this image belongs to from its path
    path_lower = img_path.replace("\\", "/").lower()
 
    if "train" in path_lower:
        embeddings["train_embeddings"].append(emb)
        embeddings["train_labels"].append(label)
        train_paths.append(img_path)
    elif "val" in path_lower:
        embeddings["val_embeddings"].append(emb)
        embeddings["val_labels"].append(label)
        val_paths.append(img_path)
    elif "test" in path_lower:
        embeddings["test_embeddings"].append(emb)
        embeddings["test_labels"].append(label)
        test_paths.append(img_path)
 
# Debug check: report split sizes
print("\nTrain:", len(train_paths))
print("Val:", len(val_paths))
print("Test:", len(test_paths))
 
# Sanity check: prevent silent crash if any split is empty
assert len(train_paths) > 0, "Train set empty"
assert len(val_paths) > 0, "Val set empty"
assert len(test_paths) > 0, "Test set empty"
 
# Save all embeddings, labels, and paths into a single .npz file
np.savez(
    embedding_file,
    train_embeddings=np.array(embeddings["train_embeddings"]),
    train_labels=np.array(embeddings["train_labels"]),
    train_paths=np.array(train_paths),
    val_embeddings=np.array(embeddings["val_embeddings"]),
    val_labels=np.array(embeddings["val_labels"]),
    val_paths=np.array(val_paths),
    test_embeddings=np.array(embeddings["test_embeddings"]),
    test_labels=np.array(embeddings["test_labels"]),
    test_paths=np.array(test_paths)
)
 
print("\nSaved:", embedding_file)
 
# Failure report
print("\nFailed images:", len(failed))
for f in failed[:10]:
    print(f)
 
# Zip the aligned images for easy download
shutil.make_archive(output_dir, "zip", output_dir)
 
# Auto-download outputs when running in Google Colab; skip silently when running locally
try:
    from google.colab import files
    files.download(output_dir + ".zip")
    files.download(embedding_file)
except ImportError:
    print("Not running in Colab - skipping auto-download.")
 
 
def visualize_pipeline(img_path):
    """
    Display a side-by-side visualization of all four pipeline stages for a
    single image: original with detection box, extracted face, aligned face,
    and final preprocessed face.
 
    Args:
        img_path: Path to an image file to run through the pipeline for display.
    """
    img = cv2.imread(img_path)
    if img is None:
        return
 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    faces = detect_faces(img_rgb)
    if len(faces) == 0:
        return
 
    rect = faces[0]
 
    # Draw the detection bounding box on a copy of the original
    img_box = img_rgb.copy()
    cv2.rectangle(
        img_box,
        (rect.left(), rect.top()),
        (rect.right(), rect.bottom()),
        (0, 255, 0), 2
    )
 
    face = extract_face(img_rgb, rect)
    aligned = align_face(face)
    processed = preprocess(aligned)
 
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
 
    ax[0].imshow(img_box)
    ax[0].set_title("Detection")
    ax[0].axis("off")
 
    ax[1].imshow(face)
    ax[1].set_title("Extraction")
    ax[1].axis("off")
 
    ax[2].imshow(aligned)
    ax[2].set_title("Alignment")
    ax[2].axis("off")
 
    ax[3].imshow(processed)
    ax[3].set_title("Preprocessed")
    ax[3].axis("off")
 
    plt.show()
 
 
# Show 5 random sample pipeline visualizations (required for deliverable)
print("\nSample Pipeline Results:")
samples = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))
for p in samples:
    visualize_pipeline(p)