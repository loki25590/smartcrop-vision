import os
import uuid
import torch
import numpy as np

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from transformers import (
    AutoImageProcessor,
    AutoConfig,
    ViTForImageClassification
)


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
# MODEL CONFIG
# ============================================================

MODEL_NAME = "google/vit-base-patch16-224"

CHECKPOINT = "checkpoint.pth"


# ============================================================
# IMAGE PROCESSOR
# ============================================================

print("Loading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# MODEL
# ============================================================

print("Loading Vision Transformer...")

id2label = {
    i: label
    for i, label in enumerate(classes)
}

label2id = {
    label: i
    for i, label in enumerate(classes)
}


# ------------------------------------------------------------
# Create configuration with 38 classes
# ------------------------------------------------------------

config = AutoConfig.from_pretrained(
    MODEL_NAME
)

config.num_labels = len(classes)
config.id2label = id2label
config.label2id = label2id


# ------------------------------------------------------------
# Load pretrained ViT architecture
# ------------------------------------------------------------

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    config=config,
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


# ------------------------------------------------------------
# Load our trained 38-class weights
# ------------------------------------------------------------

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")

print("======================================")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SmartCrop Vision API",
    description="AI Plant Disease Detection using Vision Transformer",
    version="2.0"
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
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
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
        "version": "2.0"
    }


# ============================================================
# DISEASE INFORMATION
# ============================================================

def get_disease_info(disease):

    disease_info = {

        # ----------------------------------------------------
        # APPLE
        # ----------------------------------------------------

        "Apple___Apple_scab": {
            "description": "A fungal disease that causes olive-green to brown spots on apple leaves and fruit.",
            "cause": "Fungus",
            "treatment": "Remove infected leaves and fruit and use an appropriate fungicide according to agricultural guidance.",
            "prevention": "Maintain good airflow, remove fallen leaves, avoid excessive leaf moisture, and use resistant varieties when available."
        },

        "Apple___Black_rot": {
            "description": "A fungal disease that causes dark lesions on apple leaves and fruit.",
            "cause": "Fungus",
            "treatment": "Remove infected plant material and apply suitable disease management treatment.",
            "prevention": "Remove dead wood and fallen fruit, improve airflow, and maintain good orchard hygiene."
        },

        "Apple___Cedar_apple_rust": {
            "description": "A fungal disease that causes yellow or orange spots on apple leaves.",
            "cause": "Fungus",
            "treatment": "Remove infected material and use an appropriate fungicide when recommended.",
            "prevention": "Monitor plants regularly, improve airflow, and manage nearby alternate hosts when appropriate."
        },

        "Apple___healthy": {
            "description": "The apple leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitor the plant.",
            "prevention": "Maintain proper watering, nutrition, sunlight, airflow, and regular inspection."
        },

        # ----------------------------------------------------
        # BLUEBERRY
        # ----------------------------------------------------

        "Blueberry___healthy": {
            "description": "The blueberry leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitor the plant.",
            "prevention": "Maintain suitable soil conditions, irrigation, sunlight, and regular crop monitoring."
        },

        # ----------------------------------------------------
        # CHERRY
        # ----------------------------------------------------

        "Cherry___Powdery_mildew": {
            "description": "A fungal disease that produces a white powdery coating on leaves and young plant tissue.",
            "cause": "Fungus",
            "treatment": "Remove severely affected leaves and use an appropriate fungicide if recommended.",
            "prevention": "Improve air circulation, avoid excessive humidity around foliage, and maintain adequate plant spacing."
        },

        "Cherry___healthy": {
            "description": "The cherry leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitor the plant.",
            "prevention": "Maintain good airflow, watering, nutrition, and regular inspection."
        },

        # ----------------------------------------------------
        # CORN
        # ----------------------------------------------------

        "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
            "description": "A fungal disease that causes gray or brown rectangular lesions on corn leaves.",
            "cause": "Fungus",
            "treatment": "Use suitable disease management practices and fungicides when recommended.",
            "prevention": "Rotate crops, manage crop residue, use resistant varieties, and monitor fields regularly."
        },

        "Corn___Common_rust": {
            "description": "A fungal disease that produces reddish-brown rust-like pustules on corn leaves.",
            "cause": "Fungus",
            "treatment": "Monitor disease development and use an appropriate fungicide when recommended.",
            "prevention": "Use resistant varieties, maintain good field management, and monitor plants regularly."
        },

        "Corn___Northern_Leaf_Blight": {
            "description": "A fungal disease that produces long gray-green or brown lesions on corn leaves.",
            "cause": "Fungus",
            "treatment": "Remove heavily infected material and use appropriate disease management practices.",
            "prevention": "Use resistant varieties, rotate crops, manage plant debris, and maintain good field conditions."
        },

        "Corn___healthy": {
            "description": "The corn leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal crop care and monitor the plant.",
            "prevention": "Maintain proper irrigation, nutrition, spacing, and regular crop monitoring."
        },

        # ----------------------------------------------------
        # GRAPE
        # ----------------------------------------------------

        "Grape___Black_rot": {
            "description": "A fungal disease that causes dark lesions on grape leaves and can damage developing fruit.",
            "cause": "Fungus",
            "treatment": "Remove infected plant material and use suitable fungicide treatment when recommended.",
            "prevention": "Improve airflow, remove infected debris, and avoid prolonged leaf wetness."
        },

        "Grape___Esca_(Black_Measles)": {
            "description": "A complex grapevine disease that can cause leaf symptoms and dark spotting on fruit.",
            "cause": "Fungal disease complex",
            "treatment": "Remove severely affected plant material and seek professional vineyard management advice.",
            "prevention": "Use healthy planting material, maintain vineyard hygiene, and monitor vines regularly."
        },

        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
            "description": "A fungal disease that produces spots and blight symptoms on grape leaves.",
            "cause": "Fungus",
            "treatment": "Remove infected leaves and use an appropriate fungicide when necessary.",
            "prevention": "Improve air circulation, avoid excessive moisture, and maintain vineyard hygiene."
        },

        "Grape___healthy": {
            "description": "The grape leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitoring.",
            "prevention": "Maintain good vineyard hygiene, airflow, irrigation, and nutrition."
        },

        # ----------------------------------------------------
        # ORANGE
        # ----------------------------------------------------

        "Orange___Haunglongbing_(Citrus_greening)": {
            "description": "A serious citrus disease that can cause yellowing leaves, poor fruit development, and declining tree health.",
            "cause": "Bacterial pathogen",
            "treatment": "Remove affected trees when recommended and manage insect vectors under agricultural guidance.",
            "prevention": "Use certified planting material and monitor and manage disease-carrying insect vectors."
        },

        # ----------------------------------------------------
        # PEACH
        # ----------------------------------------------------

        "Peach___Bacterial_spot": {
            "description": "A bacterial disease that causes dark spots and lesions on peach leaves and fruit.",
            "cause": "Bacteria",
            "treatment": "Remove severely affected material and use appropriate bacterial disease management practices.",
            "prevention": "Improve airflow, avoid excessive leaf wetness, and use resistant varieties where available."
        },

        "Peach___healthy": {
            "description": "The peach leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitoring.",
            "prevention": "Maintain proper watering, nutrition, sunlight, airflow, and crop hygiene."
        },

        # ----------------------------------------------------
        # PEPPER
        # ----------------------------------------------------

        "Pepper_bell___Bacterial_spot": {
            "description": "A bacterial disease that causes dark spots and lesions on pepper leaves.",
            "cause": "Bacteria",
            "treatment": "Remove affected plant material and follow appropriate bacterial disease management practices.",
            "prevention": "Avoid overhead watering, improve airflow, use healthy seeds, and maintain field hygiene."
        },

        "Pepper_bell___healthy": {
            "description": "The pepper leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitoring.",
            "prevention": "Maintain good irrigation, nutrition, sunlight, airflow, and regular inspection."
        },

        # ----------------------------------------------------
        # POTATO
        # ----------------------------------------------------

        "Potato___Early_blight": {
            "description": "A fungal disease that produces dark circular spots on potato leaves and can reduce plant growth.",
            "cause": "Fungus",
            "treatment": "Remove affected leaves, improve air circulation, and use an appropriate fungicide if necessary.",
            "prevention": "Avoid overhead irrigation, remove infected plant debris, maintain spacing, and practice crop rotation."
        },

        "Potato___Late_blight": {
            "description": "A serious disease that causes dark water-soaked lesions on potato leaves and stems.",
            "cause": "Fungus-like pathogen",
            "treatment": "Remove infected plant material and use appropriate disease management according to agricultural guidance.",
            "prevention": "Avoid prolonged leaf wetness, provide good airflow, use healthy planting material, and monitor plants frequently."
        },

        "Potato___healthy": {
            "description": "The potato leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal crop care and monitoring.",
            "prevention": "Maintain proper irrigation, nutrition, soil conditions, and regular inspection."
        },

        # ----------------------------------------------------
        # RASPBERRY
        # ----------------------------------------------------

        "Raspberry___healthy": {
            "description": "The raspberry leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitoring.",
            "prevention": "Maintain good airflow, irrigation, nutrition, and regular crop inspection."
        },

        # ----------------------------------------------------
        # SOYBEAN
        # ----------------------------------------------------

        "Soybean___healthy": {
            "description": "The soybean leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal crop care and monitoring.",
            "prevention": "Maintain appropriate irrigation, nutrition, spacing, and field monitoring."
        },

        # ----------------------------------------------------
        # SQUASH
        # ----------------------------------------------------

        "Squash___Powdery_mildew": {
            "description": "A fungal disease that produces a white powdery coating on squash leaves.",
            "cause": "Fungus",
            "treatment": "Remove severely infected leaves and use an appropriate fungicide when recommended.",
            "prevention": "Improve air circulation, avoid excessive humidity, and maintain adequate plant spacing."
        },

        # ----------------------------------------------------
        # STRAWBERRY
        # ----------------------------------------------------

        "Strawberry___Leaf_scorch": {
            "description": "A fungal disease that causes purple or brown spots and scorched areas on strawberry leaves.",
            "cause": "Fungus",
            "treatment": "Remove infected leaves, improve air circulation, and avoid excessive leaf moisture.",
            "prevention": "Avoid overhead watering, maintain good spacing between plants, remove infected leaves early, and keep the growing area clean."
        },

        "Strawberry___healthy": {
            "description": "The strawberry leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitor the plant.",
            "prevention": "Maintain proper watering, sunlight, nutrition, airflow, and regular monitoring."
        },

        # ----------------------------------------------------
        # TOMATO
        # ----------------------------------------------------

        "Tomato___Bacterial_spot": {
            "description": "A bacterial disease that causes small dark spots on tomato leaves and fruit.",
            "cause": "Bacteria",
            "treatment": "Remove affected plant material and follow appropriate bacterial disease management practices.",
            "prevention": "Avoid overhead watering, improve airflow, use healthy seeds, and maintain good crop hygiene."
        },

        "Tomato___Early_blight": {
            "description": "A fungal disease that causes dark concentric spots on tomato leaves.",
            "cause": "Fungus",
            "treatment": "Remove affected leaves, improve airflow, and apply an appropriate fungicide when necessary.",
            "prevention": "Avoid overhead irrigation, remove plant debris, maintain spacing, and rotate crops."
        },

        "Tomato___Late_blight": {
            "description": "A destructive disease that causes dark lesions on tomato leaves and can spread rapidly in humid conditions.",
            "cause": "Fungus-like pathogen",
            "treatment": "Remove infected material and seek appropriate agricultural treatment quickly.",
            "prevention": "Improve airflow, avoid prolonged leaf moisture, monitor plants frequently, and use healthy planting material."
        },

        "Tomato___Leaf_Mold": {
            "description": "A fungal disease that can produce yellow areas on the upper leaf surface and mold growth underneath.",
            "cause": "Fungus",
            "treatment": "Remove affected leaves and improve ventilation around the plants.",
            "prevention": "Reduce humidity, improve airflow, avoid wetting foliage, and maintain proper spacing."
        },

        "Tomato___Septoria_leaf_spot": {
            "description": "A fungal disease that produces small circular spots on tomato leaves and can cause premature leaf loss.",
            "cause": "Fungus",
            "treatment": "Remove infected leaves, improve airflow, and use an appropriate fungicide when recommended.",
            "prevention": "Avoid overhead watering, remove infected debris, maintain plant spacing, and rotate crops."
        },

        "Tomato___Spider_mites Two-spotted_spider_mite": {
            "description": "A pest infestation that can cause yellow speckling, leaf damage, and reduced plant vigor.",
            "cause": "Spider mites",
            "treatment": "Use appropriate pest management and remove heavily affected leaves when necessary.",
            "prevention": "Monitor leaves regularly, maintain adequate plant hydration, and encourage beneficial insects where appropriate."
        },

        "Tomato___Target_Spot": {
            "description": "A fungal disease that produces circular target-like spots on tomato leaves and fruit.",
            "cause": "Fungus",
            "treatment": "Remove infected plant material and use suitable fungicide treatment when recommended.",
            "prevention": "Improve airflow, avoid overhead irrigation, remove crop debris, and maintain proper spacing."
        },

        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
            "description": "A viral disease that causes yellowing, curling, and stunted growth of tomato leaves.",
            "cause": "Virus",
            "treatment": "Remove severely infected plants and manage insect vectors according to agricultural guidance.",
            "prevention": "Control whiteflies, use healthy planting material, and remove infected plants promptly."
        },

        "Tomato___Tomato_mosaic_virus": {
            "description": "A viral disease that can cause mottled or mosaic patterns on tomato leaves and reduced plant growth.",
            "cause": "Virus",
            "treatment": "Remove infected plants and maintain strict sanitation practices.",
            "prevention": "Use clean tools and planting material, remove infected plants, and avoid spreading plant sap between plants."
        },

        "Tomato___healthy": {
            "description": "The tomato leaf appears healthy with no detected disease.",
            "cause": "None detected",
            "treatment": "Continue normal plant care and monitor the plant.",
            "prevention": "Maintain proper watering, sunlight, nutrition, spacing, and regular inspection."
        }
    }

    return disease_info.get(
        disease,
        {
            "description": "Information for this disease is not available yet.",
            "cause": "Unknown",
            "treatment": "Consult an agricultural expert for an appropriate diagnosis and treatment.",
            "prevention": "Monitor the crop regularly, maintain good field hygiene, and consult an agricultural expert if symptoms worsen."
        }
    )


# ============================================================
# DISEASE SEVERITY
# ============================================================

def get_severity(
    disease_name,
    confidence
):

    if disease_name.lower() == "healthy":
        return "Healthy"

    if confidence >= 80:
        return "High"

    if confidence >= 50:
        return "Moderate"

    return "Low"


# ============================================================
# HEALTH STATUS
# ============================================================

def get_health_status(
    disease_name,
    confidence
):

    if disease_name.lower() == "healthy":
        return "Healthy"

    if confidence >= 50:
        return "Diseased"

    return "Needs Review"


# ============================================================
# AI EXPLANATION
# ============================================================

def get_ai_explanation(
    disease_name
):

    if disease_name.lower() == "healthy":

        return (
            "The Vision Transformer did not identify strong "
            "attention patterns associated with a known plant disease. "
            "The leaf appears visually similar to healthy examples "
            "from the model's training data."
        )

    return (
        "The Vision Transformer concentrated its attention on "
        "visual regions of the leaf that contributed strongly to "
        "the predicted disease. The highlighted areas in the "
        "AI Focus image represent regions receiving higher model attention."
    )


# ============================================================
# CREATE AI VISION HEATMAP
# ============================================================

def create_attention_map(
    image,
    attentions
):

    print("Creating AI Vision attention heatmap...")

    # --------------------------------------------------------
    # Get final transformer layer
    # --------------------------------------------------------

    last_attention = attentions[-1]

    # Expected:
    # [batch, heads, tokens, tokens]
    # [1, 12, 197, 197]

    attention = last_attention[0]

    # --------------------------------------------------------
    # Average attention heads
    # --------------------------------------------------------

    attention = attention.mean(
        dim=0
    )

    # --------------------------------------------------------
    # CLS token -> image patches
    # --------------------------------------------------------

    cls_attention = attention[
        0,
        1:
    ]

    # --------------------------------------------------------
    # Calculate patch dimensions
    # --------------------------------------------------------

    patch_count = cls_attention.shape[0]

    patch_size = int(
        np.sqrt(
            patch_count
        )
    )

    if patch_size * patch_size != patch_count:

        raise ValueError(
            f"Unexpected number of image patches: {patch_count}"
        )

    # --------------------------------------------------------
    # Reshape attention into image grid
    # --------------------------------------------------------

    attention_map = cls_attention.reshape(
        patch_size,
        patch_size
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    attention_map = (
        attention_map -
        attention_map.min()
    )

    attention_map = (
        attention_map /
        (
            attention_map.max()
            + 1e-8
        )
    )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    attention_map = (
        attention_map
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Convert attention to image
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            attention_map * 255
        )
    )

    # --------------------------------------------------------
    # Resize to original image size
    # --------------------------------------------------------

    heatmap_image = heatmap_image.resize(
        image.size,
        Image.Resampling.BICUBIC
    )

    heatmap_array = np.array(
        heatmap_image
    )

    # --------------------------------------------------------
    # Create colored heatmap
    # --------------------------------------------------------

    heatmap_rgb = np.zeros(
        (
            heatmap_array.shape[0],
            heatmap_array.shape[1],
            3
        ),
        dtype=np.uint8
    )

    heatmap_rgb[:, :, 0] = (
        heatmap_array
    )

    heatmap_rgb[:, :, 1] = (
        heatmap_array * 0.25
    ).astype(np.uint8)

    heatmap_rgb[:, :, 2] = (
        255 - heatmap_array
    )

    heatmap = Image.fromarray(
        heatmap_rgb
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        original,
        heatmap,
        alpha=0.45
    )

    return overlay


# ============================================================
# PREDICTION
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
        # MODEL PREDICTION
        # ====================================================

        print(
            "Running Vision Transformer..."
        )

        with torch.no_grad():

            outputs = model(
                pixel_values=pixel_values,
                output_attentions=True
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

        print(
            "Attention layers:",
            len(outputs.attentions)
        )

        # ====================================================
        # CONVERT RESULTS
        # ====================================================

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

            disease = classes[index]

            # ------------------------------------------------
            # Split class
            # ------------------------------------------------

            if "___" in disease:

                plant, disease_name = (
                    disease.split(
                        "___",
                        1
                    )
                )

            else:

                plant = "Unknown"
                disease_name = disease

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = round(
                probability * 100,
                2
            )

            # ------------------------------------------------
            # Disease information
            # ------------------------------------------------

            info = get_disease_info(
                disease
            )

            # ------------------------------------------------
            # Severity
            # ------------------------------------------------

            severity = get_severity(
                disease_name,
                confidence
            )

            # ------------------------------------------------
            # Health status
            # ------------------------------------------------

            health_status = get_health_status(
                disease_name,
                confidence
            )

            # ------------------------------------------------
            # AI explanation
            # ------------------------------------------------

            ai_explanation = get_ai_explanation(
                disease_name
            )

            # ------------------------------------------------
            # Prediction object
            # ------------------------------------------------

            predictions.append({

                "disease": disease,

                "plant": plant,

                "disease_name": disease_name,

                "confidence": confidence,

                "severity": severity,

                "health_status": health_status,

                "ai_explanation": ai_explanation,

                "info": info

            })

        # ====================================================
        # GENERATE AI VISION
        # ====================================================

        print(
            "Generating AI Vision heatmap..."
        )

        attention_image = create_attention_map(
            image,
            outputs.attentions
        )

        # ====================================================
        # UNIQUE OUTPUT FILE
        # ====================================================

        filename = (
            "attention_"
            + uuid.uuid4().hex
            + ".jpg"
        )

        output_path = os.path.join(
            "outputs",
            filename
        )

        # ====================================================
        # SAVE HEATMAP
        # ====================================================

        attention_image.save(
            output_path,
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
            "======================================"
        )

        print(
            "Prediction error:"
        )

        print(
            str(e)
        )

        print(
            "======================================"
        )

        return {

            "success": False,

            "error": str(e)

        }