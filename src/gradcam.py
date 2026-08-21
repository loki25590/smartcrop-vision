import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT = "checkpoint.pth"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Grad-CAM Device:", device)


# ============================================================
# CLASSES
# ============================================================

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
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
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ============================================================
# PROCESSOR
# ============================================================

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# MODEL
# ============================================================

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=38,
    ignore_mismatched_sizes=True,
    attn_implementation="eager"
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
# GENERATE ATTENTION VISUALIZATION
# ============================================================

def generate_gradcam(
    image: Image.Image,
    filename: str = "gradcam_result.jpg"
):

    image = image.convert("RGB")

    inputs = processor(
        image,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)


    # ========================================================
    # MODEL FORWARD
    # ========================================================

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values,
            output_attentions=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    confidence, predicted_id = probabilities.max(
        dim=-1
    )

    predicted_id = predicted_id.item()

    confidence = confidence.item()


    # ========================================================
    # ATTENTION
    # ========================================================

    attentions = outputs.attentions

    if attentions is None:

        raise RuntimeError(
            "Model did not return attention maps."
        )


    # ========================================================
    # ATTENTION ROLLOUT
    # ========================================================

    result = torch.eye(
        attentions[0].shape[-1],
        device=device
    )


    for attention in attentions:

        # Average heads
        attention = attention.mean(
            dim=1
        )[0]


        # Residual connection
        attention = attention + torch.eye(
            attention.shape[-1],
            device=device
        )


        # Normalize
        attention = attention / attention.sum(
            dim=-1,
            keepdim=True
        )


        # Rollout
        result = torch.matmul(
            attention,
            result
        )


    # ========================================================
    # CLASS TOKEN ATTENTION
    # ========================================================

    mask = result[0, 1:]

    num_patches = mask.shape[0]

    grid_size = int(
        np.sqrt(num_patches)
    )


    if grid_size * grid_size != num_patches:

        raise RuntimeError(
            f"Unexpected number of patches: {num_patches}"
        )


    mask = mask.reshape(
        grid_size,
        grid_size
    )


    mask = mask.detach().cpu().numpy()


    # ========================================================
    # NORMALIZE
    # ========================================================

    mask = mask - mask.min()


    if mask.max() > 0:

        mask = mask / mask.max()


    # ========================================================
    # RESIZE HEATMAP
    # ========================================================

    heatmap = Image.fromarray(
        np.uint8(mask * 255)
    )


    heatmap = heatmap.resize(
        image.size,
        Image.Resampling.BILINEAR
    )


    heatmap = np.array(
        heatmap
    ) / 255.0


    # ========================================================
    # OUTPUT PATH
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    # Original
    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(image)

    plt.title(
        f"Prediction:\n{classes[predicted_id]}"
    )

    plt.axis("off")


    # Attention
    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(image)

    plt.imshow(
        heatmap,
        alpha=0.5
    )

    plt.title(
        f"AI Attention\n"
        f"{confidence * 100:.2f}% confidence"
    )

    plt.axis("off")


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    return {
        "path": output_path,
        "predicted_id": predicted_id,
        "disease": classes[predicted_id],
        "confidence": round(
            confidence * 100,
            2
        )
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    IMAGE_PATH = "test.jpg"

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")


    result = generate_gradcam(
        image,
        "gradcam_test.jpg"
    )


    print()
    print(
        "Predicted disease:",
        result["disease"]
    )

    print(
        "Confidence:",
        result["confidence"],
        "%"
    )

    print(
        "Saved visualization:",
        result["path"]
    )