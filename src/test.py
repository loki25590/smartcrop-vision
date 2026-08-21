import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from dataset import PlantDiseaseDataset
from dataset import test_samples
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=38,
    ignore_mismatched_sizes=True
)
checkpoint = torch.load(
    "checkpoint.pth",
    map_location="cpu"
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

model.eval()
test_dataset = PlantDiseaseDataset(test_samples)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
correct = 0
total = 0

with torch.no_grad():

    for images, labels in tqdm(
    test_loader,
    desc="Testing"
):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images)

        predictions = outputs.logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("Test Accuracy:", accuracy)