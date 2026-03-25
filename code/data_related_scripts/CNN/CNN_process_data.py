import os
import shutil

training_dir = "data/preprocessed/training"
validation_dir = "data/preprocessed/validation"
test_dir = "data/preprocessed/test"

cnn_preprocessed_dir_training = "data/processed/CNN/training/"
cnn_preprocessed_dir_validation = "data/processed/CNN/validation/"
cnn_preprocessed_dir_test = "data/processed/CNN/test/"

dirs = [
    (training_dir, cnn_preprocessed_dir_training),
    (validation_dir, cnn_preprocessed_dir_validation),
    (test_dir, cnn_preprocessed_dir_test)
]

for current_inital_dir, current_final_dir in dirs:
    
    for filename in os.listdir(current_inital_dir):
        if not filename.endswith(".jpg"):
            continue

        parts = filename.split("_", 1)
        if len(parts) < 2:
            continue

        class_name = parts[1].replace(".jpg", "")
        class_folder = os.path.join(current_final_dir, class_name)

        os.makedirs(class_folder, exist_ok=True)

        old_path = os.path.join(current_inital_dir, filename)
        new_path = os.path.join(class_folder, filename)

        shutil.copy(old_path, new_path)