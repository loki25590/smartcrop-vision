import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD CONFUSION MATRIX
# ============================================================

cm = np.load("confusion_matrix.npy")


# ============================================================
# CLASS NAMES
# ============================================================

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper___Bacterial_spot",
    "Pepper___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___mosaic_virus",
    "Tomato___healthy",
]


# ============================================================
# TOP CONFUSIONS
# ============================================================

print("\n==============================")
print("TOP MODEL CONFUSIONS")
print("==============================")

confusions = []

for actual in range(len(classes)):

    for predicted in range(len(classes)):

        if actual == predicted:
            continue

        count = cm[actual][predicted]

        if count > 0:

            confusions.append(
                (
                    count,
                    classes[actual],
                    classes[predicted]
                )
            )


confusions.sort(
    reverse=True
)


for count, actual, predicted in confusions[:15]:

    print(
        f"{actual}"
        f"  →  "
        f"{predicted}"
        f" : {count}"
    )


# ============================================================
# WORST CLASSES
# ============================================================

print("\n==============================")
print("WORST CLASSES")
print("==============================")


for i, class_name in enumerate(classes):

    total = cm[i].sum()

    correct = cm[i][i]

    recall = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:55s}"
        f" Recall: {recall * 100:6.2f}%"
        f"  Samples: {total}"
    )


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(18, 16)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "SmartCrop Vision - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.colorbar()

plt.xticks(
    range(len(classes)),
    range(len(classes)),
    rotation=90
)

plt.yticks(
    range(len(classes)),
    range(len(classes))
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=200
)

print(
    "\nSaved:"
    " confusion_matrix.png"
)
