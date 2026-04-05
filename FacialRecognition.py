import os
from PIL import Image
from tqdm import tqdm

# repo_dir = "ITEC-612-Facial-Recognition"
# os.chdir(repo_dir)

dataset_dir = "dataset"
processed_dir = "processed_data"
os.makedirs(processed_dir, exist_ok=True)

target_size = (224,224)
mode = "RGB"

def preprocess_and_save(input_dir, output_dir):
    for split in os.listdir(input_dir): # train, val, test
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
    
        for person in os.listdir(split_input_dir):
            person_input_dir = os.path.join(split_input_dir, person)
            person_output_dir = os.path.join(split_output_dir, person)
            os.makedirs(person_output_dir, exist_ok=True)

            for img_file in tqdm(os.listdir(person_input_dir), desc=f"Processing {split}/{person}"):
                img_path = os.path.join(person_input_dir, img_file)
                try:
                    img = Image.open(img_path).convert(mode)
                    img = img.resize(target_size)
                    save_path = os.path.join(person_output_dir, img_file)
                    img.save(save_path)
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

# Step 5 Run Preprocessing

preprocess_and_save(dataset_dir, processed_dir)
print("✅ Preprocessing complete! Check the processed_data folder.")
