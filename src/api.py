import os
import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoImageProcessor, ViTForImageClassification
from app.disease_info import get_disease_info


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

print("Device:", device)


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
# IMAGE PROCESSOR
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
# FASTAPI
# ============================================================

app = FastAPI(
    title="SmartCrop Vision API",
    description="Plant disease classification API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVE OUTPUT IMAGES
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "project": "SmartCrop Vision",
        "status": "running"
    }


# ============================================================
# ATTENTION ROLLOUT
# ============================================================

def generate_attention_heatmap(
    image,
    pixel_values,
    filename
):

    # --------------------------------------------------------
    # Get attention maps
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values,
            output_attentions=True
        )

    attentions = outputs.attentions

    if attentions is None:

        raise RuntimeError(
            "Model did not return attention maps."
        )


    # --------------------------------------------------------
    # Attention rollout
    # --------------------------------------------------------

    result = torch.eye(
        attentions[0].shape[-1],
        device=device
    )


    for attention in attentions:

        # Average attention across heads
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


    # --------------------------------------------------------
    # Class token attention
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Normalize heatmap
    # --------------------------------------------------------

    mask = mask - mask.min()


    if mask.max() > 0:

        mask = mask / mask.max()


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap = Image.fromarray(
        np.uint8(mask * 255)
    )


    heatmap = heatmap.resize(
        image.size,
        Image.Resampling.BILINEAR
    )


    heatmap_array = np.array(
        heatmap
    ) / 255.0


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt


    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(image)

    plt.imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")

    plt.tight_layout(
        pad=0
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


    return output_path


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        file.file
    ).convert("RGB")


    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    inputs = processor(
        image,
        return_tensors="pt"
    )


    pixel_values = inputs[
        "pixel_values"
    ].to(device)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )


        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )


        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )


    top_probabilities = (
        top_probabilities[0]
        .cpu()
        .tolist()
    )

    top_indices = (
        top_indices[0]
        .cpu()
        .tolist()
    )


    predictions = []


    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        disease = classes[index]

        plant, disease_name = disease.split("___", 1)

        info = get_disease_info(disease)

        predictions.append({

            "disease": disease,

            "plant": plant,

            "disease_name": disease_name,

            "confidence": round(
                probability * 100,
                2
            ),

            "info": info

        })


    # --------------------------------------------------------
    # Generate unique heatmap filename
    # --------------------------------------------------------

    heatmap_filename = (
        "heatmap_"
        + str(torch.randint(
            100000,
            999999,
            (1,)
        ).item())
        + ".jpg"
    )


    # --------------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------------

    generate_attention_heatmap(
        image,
        pixel_values,
        heatmap_filename
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "top_prediction": predictions[0],

        "predictions": predictions,

        "heatmap": (
            "/outputs/"
            + heatmap_filename
        )

    }