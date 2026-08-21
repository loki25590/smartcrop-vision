import os
import uuid
import torch
import numpy as np

from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT = BASE_DIR / "checkpoint.pth"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("SmartCrop Vision")
print("======================================")
print("Device:", device)
print("Checkpoint:", CHECKPOINT)
print("Output directory:", OUTPUT_DIR)


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
# MODEL
# ============================================================

MODEL_NAME = "google/vit-base-patch16-224"

print()
print("Loading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading Vision Transformer...")

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(classes),
    ignore_mismatched_sizes=True,
    attn_implementation="eager"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

if "model_state_dict" in checkpoint:

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")
print("Number of classes:", len(classes))


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SmartCrop Vision API",
    description="Plant Disease Detection using Vision Transformer",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# STATIC OUTPUT DIRECTORY
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "project": "SmartCrop Vision",
        "status": "running",
        "model": "Vision Transformer",
        "classes": len(classes)
    }


# ============================================================
# ATTENTION ROLLOUT
# ============================================================

def create_attention_map(
    image,
    attentions
):

    """
    Creates an AI Vision attention heatmap.

    Uses attention rollout across all ViT layers.

    ViT-base:
        12 transformer layers
        12 attention heads
        197 tokens
        196 image patches
        14 x 14 patch grid
    """

    print("Creating attention rollout...")

    if attentions is None:

        raise RuntimeError(
            "Model did not return attention tensors."
        )


    # --------------------------------------------------------
    # Start with identity matrix
    # --------------------------------------------------------

    num_tokens = attentions[0].shape[-1]

    rollout = torch.eye(
        num_tokens,
        device=attentions[0].device
    )


    # --------------------------------------------------------
    # Process every transformer layer
    # --------------------------------------------------------

    for layer_attention in attentions:

        # Shape:
        #
        # [batch, heads, tokens, tokens]
        

        attention = layer_attention[0]


        # ----------------------------------------------------
        # Average attention heads
        # ----------------------------------------------------

        attention = attention.mean(
            dim=0
        )


        # ----------------------------------------------------
        # Add residual connection
        # ----------------------------------------------------

        identity = torch.eye(
            attention.shape[-1],
            device=attention.device
        )

        attention = attention + identity


        # ----------------------------------------------------
        # Normalize rows
        # ----------------------------------------------------

        attention = attention / (
            attention.sum(
                dim=-1,
                keepdim=True
            ) + 1e-8
        )


        # ----------------------------------------------------
        # Rollout
        # ----------------------------------------------------

        rollout = torch.matmul(
            attention,
            rollout
        )


    # ========================================================
    # CLS TOKEN -> IMAGE PATCHES
    # ========================================================

    cls_attention = rollout[
        0,
        1:
    ]


    # ========================================================
    # 196 PATCHES -> 14 x 14
    # ========================================================

    patch_size = int(
        np.sqrt(
            cls_attention.shape[0]
        )
    )

    if patch_size * patch_size != cls_attention.shape[0]:

        raise RuntimeError(
            "Unexpected ViT attention patch size."
        )


    attention_map = cls_attention.reshape(
        patch_size,
        patch_size
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    attention_map = (
        attention_map -
        attention_map.min()
    )

    attention_map = (
        attention_map /
        (attention_map.max() + 1e-8)
    )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    attention_array = (
        attention_map
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # RESIZE ATTENTION MAP
    # ========================================================

    heatmap_gray = Image.fromarray(
        np.uint8(
            attention_array * 255
        )
    )

    heatmap_gray = heatmap_gray.resize(
        image.size,
        Image.Resampling.BICUBIC
    )


    heatmap_array = np.array(
        heatmap_gray
    ).astype(
        np.float32
    ) / 255.0


    # ========================================================
    # CREATE COLORED HEATMAP
    # ========================================================

    # Blue -> low attention
    # Red  -> high attention

    heatmap_rgb = np.zeros(
        (
            heatmap_array.shape[0],
            heatmap_array.shape[1],
            3
        ),
        dtype=np.uint8
    )


    # Red
    heatmap_rgb[:, :, 0] = (
        heatmap_array * 255
    ).astype(np.uint8)


    # Green
    heatmap_rgb[:, :, 1] = (
        heatmap_array * 100
    ).astype(np.uint8)


    # Blue
    heatmap_rgb[:, :, 2] = (
        (1.0 - heatmap_array) * 255
    ).astype(np.uint8)


    heatmap = Image.fromarray(
        heatmap_rgb
    ).convert("RGB")


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original = image.convert(
        "RGB"
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = Image.blend(
        original,
        heatmap,
        alpha=0.50
    )


    return overlay


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        print()
        print("======================================")
        print("New image received")
        print("======================================")


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(
            file.file
        ).convert("RGB")

        print(
            "Image size:",
            image.size
        )


        # ====================================================
        # PROCESS IMAGE
        # ====================================================

        inputs = processor(
            image,
            return_tensors="pt"
        )


        pixel_values = inputs[
            "pixel_values"
        ].to(device)


        # ====================================================
        # MODEL
        # ====================================================

        print(
            "Running Vision Transformer..."
        )

        with torch.no_grad():

            outputs = model(
                pixel_values=pixel_values,
                output_attentions=True
            )


        # ====================================================
        # CHECK ATTENTION
        # ====================================================

        if outputs.attentions is None:

            raise RuntimeError(
                "Vision Transformer did not return attention."
            )


        print(
            "Attention layers:",
            len(outputs.attentions)
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )


        # ====================================================
        # TOP 3
        # ====================================================

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


        # ====================================================
        # CREATE PREDICTIONS
        # ====================================================

        predictions = []


        for probability, index in zip(
            top_probabilities,
            top_indices
        ):

            disease_name = classes[index]


            # Split plant and disease

            if "___" in disease_name:

                plant, disease = disease_name.split(
                    "___",
                    1
                )

            else:

                plant = "Unknown"
                disease = disease_name


            predictions.append({

                "disease": disease_name,

                "plant": plant,

                "disease_name": disease,

                "confidence": round(
                    probability * 100,
                    2
                )

            })


        # ====================================================
        # AI VISION
        # ====================================================

        print(
            "Generating AI Vision heatmap..."
        )

        attention_image = create_attention_map(
            image,
            outputs.attentions
        )


        # ====================================================
        # UNIQUE FILE NAME
        # ====================================================

        filename = (
            f"attention_{uuid.uuid4().hex}.jpg"
        )


        output_path = (
            OUTPUT_DIR /
            filename
        )


        # ====================================================
        # SAVE
        # ====================================================

        attention_image.save(
            output_path,
            format="JPEG",
            quality=95
        )


        print(
            "AI Vision saved:"
        )

        print(
            output_path
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "top_prediction":
                predictions[0],

            "predictions":
                predictions,

            "ai_vision_url":
                f"/outputs/{filename}"

        }


    except Exception as e:

        print()
        print(
            "Prediction error:"
        )

        print(
            repr(e)
        )


        return {

            "success": False,

            "error": str(e)

        }


# ============================================================
# END
# ============================================================
