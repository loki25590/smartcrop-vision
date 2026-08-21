import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "checkpoint.pth"

TEST_DIR = "data/plantvillage/data_distribution_for_SVM/test"

MAPPING_FILE = (
    "data/plantvillage/data_distribution_for_SVM/test_mapping.txt"
)

MODEL_NAME = "google/vit-base-patch16-224"

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CLASS MAPPING
# ============================================================

folder_to_class = {}

with open(MAPPING_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        match = re.search(
            r"SVM/test/(\d+)/",
            line
        )

        if not match:
            continue

        folder_id = int(match.group(1))

        source = line.split("\t")[0]

        parts = source.split("/")

        class_name = parts[2]

        folder_to_class[folder_id] = class_name


print(
    "Number of classes found:",
    len(folder_to_class)
)


# ============================================================
# MODEL
# ============================================================

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=38,
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# COLLECT TEST IMAGES
# ============================================================

image_paths = []
true_labels = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
)

for folder_name in sorted(
    os.listdir(TEST_DIR),
    key=lambda x: int(x) if x.isdigit() else 999
):

    folder_path = os.path.join(
        TEST_DIR,
        folder_name
    )

    if not os.path.isdir(folder_path):
        continue

    if not folder_name.isdigit():
        continue

    folder_id = int(folder_name)

    for filename in os.listdir(folder_path):

        if not filename.endswith(valid_extensions):
            continue

        image_paths.append(
            os.path.join(
                folder_path,
                filename
            )
        )

        true_labels.append(folder_id)


print(
    "Total test images:",
    len(image_paths)
)


# ============================================================
# PREDICTION
# ============================================================

predicted_labels = []

total = len(image_paths)

for start in range(
    0,
    total,
    BATCH_SIZE
):

    batch_paths = image_paths[
        start:start + BATCH_SIZE
    ]

    images = []

    valid_indices = []

    for i, path in enumerate(batch_paths):

        try:

            image = Image.open(
                path
            ).convert("RGB")

            images.append(image)

            valid_indices.append(i)

        except Exception as e:

            print(
                "Skipping:",
                path,
                "|",
                e
            )

    if not images:
        continue

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

    predicted_labels.extend(
        predictions.cpu().numpy().tolist()
    )

    if (
        start % (BATCH_SIZE * 10) == 0
        or start + BATCH_SIZE >= total
    ):

        print(
            f"Processed {min(start + BATCH_SIZE, total)}"
            f"/{total}"
        )


# ============================================================
# METRICS
# ============================================================

true_labels = np.array(
    true_labels[:len(predicted_labels)]
)

predicted_labels = np.array(
    predicted_labels
)


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

labels = sorted(
    folder_to_class.keys()
)

target_names = [
    folder_to_class.get(
        label,
        f"Class_{label}"
    )
    for label in labels
]

print("\n==============================")
print("PER-CLASS RESULTS")
print("==============================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=labels
)

np.save(
    "confusion_matrix.npy",
    cm
)

print(
    "\nConfusion matrix saved to:"
    " confusion_matrix.npy"
)

print("\nEvaluation complete.")
