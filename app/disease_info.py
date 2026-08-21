DISEASE_INFO = {

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease_name": "Leaf Scorch",
        "description": "A fungal disease that causes dark purple to brown spots on strawberry leaves.",
        "symptoms": [
            "Small dark spots appear on leaves",
            "Spots may enlarge and develop brown centers",
            "Severely affected leaves may dry out",
            "Leaf tissue can eventually become scorched"
        ],
        "cause": "Commonly associated with fungal infection and prolonged leaf wetness.",
        "management": [
            "Remove severely infected leaves",
            "Improve air circulation around plants",
            "Avoid excessive moisture on leaves",
            "Use an appropriate fungicide if recommended locally"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Keep plants properly spaced",
            "Remove infected plant debris",
            "Maintain good field sanitation"
        ]
    },

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease_name": "Early Blight",
        "description": "A fungal disease that produces dark lesions on potato leaves and can reduce plant productivity.",
        "symptoms": [
            "Dark circular spots on older leaves",
            "Concentric ring patterns may appear",
            "Leaves may turn yellow and dry",
            "Disease can progress upward through the plant"
        ],
        "cause": "Caused by the fungus Alternaria solani.",
        "management": [
            "Remove heavily infected leaves",
            "Improve air circulation",
            "Avoid prolonged leaf wetness",
            "Use appropriate fungicide when necessary"
        ],
        "prevention": [
            "Practice crop rotation",
            "Use healthy planting material",
            "Remove infected plant debris",
            "Avoid overhead irrigation"
        ]
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease_name": "Late Blight",
        "description": "A serious disease that can rapidly damage potato foliage and tubers.",
        "symptoms": [
            "Dark water-soaked lesions",
            "Brown or black areas develop on leaves",
            "Rapid leaf collapse may occur",
            "Tubers can develop brown lesions"
        ],
        "cause": "Caused by Phytophthora infestans.",
        "management": [
            "Remove severely infected plant material",
            "Avoid prolonged leaf wetness",
            "Monitor plants frequently",
            "Use appropriate disease-control products when recommended"
        ],
        "prevention": [
            "Use healthy seed potatoes",
            "Maintain good field sanitation",
            "Avoid excessive irrigation",
            "Monitor weather conditions favorable to disease"
        ]
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease_name": "Healthy",
        "description": "The model detected no obvious disease pattern in the uploaded potato leaf.",
        "symptoms": [
            "Healthy green foliage",
            "No major disease lesions detected"
        ],
        "cause": "No disease detected.",
        "management": [
            "Continue normal crop care",
            "Monitor the plant regularly"
        ],
        "prevention": [
            "Maintain good field hygiene",
            "Provide appropriate water and nutrients",
            "Regularly inspect leaves for changes"
        ]
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease_name": "Healthy",
        "description": "The model detected no obvious disease pattern in the uploaded strawberry leaf.",
        "symptoms": [
            "Healthy green leaves",
            "No major disease symptoms detected"
        ],
        "cause": "No disease detected.",
        "management": [
            "Continue normal crop care",
            "Monitor the plant regularly"
        ],
        "prevention": [
            "Maintain good air circulation",
            "Avoid excessive leaf moisture",
            "Remove damaged plant material",
            "Inspect plants regularly"
        ]
    }
}


def get_disease_info(disease):
    """
    Return information for a predicted disease.
    """

    return DISEASE_INFO.get(
        disease,
        {
            "plant": disease.split("___")[0]
            if "___" in disease else "Unknown",

            "disease_name": disease.split("___")[1]
            if "___" in disease else disease,

            "description": "Disease information is not available yet.",

            "symptoms": [],

            "cause": "Information not available.",

            "management": [],

            "prevention": []
        }
    )