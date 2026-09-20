import os
import shutil
from sklearn.model_selection import train_test_split

# Original dataset
SOURCE = r"C:\Users\palad\.cache\kagglehub\competitions\paddy-disease-classification\train_images"

# New dataset folder inside your project
DEST = r"D:\MAJOR PROJECT\dataset"

# Classes
classes = os.listdir(SOURCE)

# Create train and validation folders
for split in ["train", "val"]:
    for class_name in classes:
        os.makedirs(os.path.join(DEST, split, class_name), exist_ok=True)

# Process each class
for class_name in classes:

    source_folder = os.path.join(SOURCE, class_name)
    images = os.listdir(source_folder)

    # 80% training, 20% validation
    train_images, val_images = train_test_split(
        images,
        test_size=0.20,
        random_state=42
    )

    # Copy training images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(DEST, "train", class_name, image)
        )

    # Copy validation images
    for image in val_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(DEST, "val", class_name, image)
        )

    print(
        class_name,
        "→ Train:", len(train_images),
        "Validation:", len(val_images)
    )

print("\nDataset preparation completed!")