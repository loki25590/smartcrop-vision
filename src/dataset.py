import os

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor


DATA_PATH = "data/plantvillage/raw/segmented"

processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)


# -------------------------
# Get class folders
# -------------------------

classes = []

for item in os.listdir(DATA_PATH):

    class_path = os.path.join(DATA_PATH, item)

    if os.path.isdir(class_path):
        classes.append(item)

classes = sorted(classes)


# -------------------------
# Create class → ID mapping
# -------------------------

label_to_id = {}

for index, item in enumerate(classes):
    label_to_id[item] = index


# -------------------------
# Create (image_path, class_id) samples
# -------------------------

samples = []

for item in classes:

    class_id = label_to_id[item]
    class_path = os.path.join(DATA_PATH, item)

    images = os.listdir(class_path)

    for image in images:

        image_path = os.path.join(
            class_path,
            image
        )

        samples.append(
            (image_path, class_id)
        )


# -------------------------
# Extract labels
# -------------------------

labels = []

for sample in samples:
    labels.append(sample[1])


# -------------------------
# Train / temporary split
# 70% train
# 30% temporary
# -------------------------

train_samples, temp_samples = train_test_split(
    samples,
    test_size=0.30,
    stratify=labels,
    random_state=42
)


# -------------------------
# Validation / test split
# 15% validation
# 15% test
# -------------------------

temp_labels = []

for sample in temp_samples:
    temp_labels.append(sample[1])


val_samples, test_samples = train_test_split(
    temp_samples,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)


# -------------------------
# Dataset class
# -------------------------

class PlantDiseaseDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        processed = processor(
            image,
            return_tensors="pt"
        )

        image = processed[
            "pixel_values"
        ].squeeze(0)

        return image, label


# -------------------------
# Create datasets
# -------------------------

train_dataset = PlantDiseaseDataset(
    train_samples
)

val_dataset = PlantDiseaseDataset(
    val_samples
)

test_dataset = PlantDiseaseDataset(
    test_samples
)


# -------------------------
# Create DataLoaders
# -------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)