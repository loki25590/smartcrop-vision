from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
from torch.optim import AdamW
from transformers import AutoImageProcessor, ViTForImageClassification
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
DATA_PATH = "data/plantvillage/raw/segmented"
processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
# 1. Get class folders
classes = []

for item in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, item)

    if os.path.isdir(class_path):
        classes.append(item)

classes = sorted(classes)


# 2. Create class → ID mapping
label_to_id = {}

for index, item in enumerate(classes):
    label_to_id[item] = index


# 3. Create (image_path, class_id) samples
samples = []

for item in classes:
    class_id = label_to_id[item]
    class_path = os.path.join(DATA_PATH, item)

    images = os.listdir(class_path)

    for image in images:
        image_path = os.path.join(class_path, image)
        samples.append((image_path, class_id))


# 4. Extract labels for stratification
labels = []

for sample in samples:
    labels.append(sample[1])


# 5. First split: 70% train, 30% temporary
train_samples, temp_samples = train_test_split(
    samples,
    test_size=0.30,
    stratify=labels,
    random_state=42
)


# 6. Get labels for temporary set
temp_labels = []

for sample in temp_samples:
    temp_labels.append(sample[1])


# 7. Second split: 15% validation, 15% test
val_samples, test_samples = train_test_split(
    temp_samples,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)


class PlantDiseaseDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        processed = processor(image, return_tensors="pt")
        image = processed["pixel_values"].squeeze(0)
        return image, label


train_dataset = PlantDiseaseDataset(train_samples)
val_dataset = PlantDiseaseDataset(val_samples)
test_dataset = PlantDiseaseDataset(test_samples)
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

images, labels = next(iter(train_loader))
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(classes),
    ignore_mismatched_sizes=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

criterion = CrossEntropyLoss()

for epoch in range(1):

    model.train()

    total_loss = 0

    for batch_idx, (images, labels) in enumerate(
    tqdm(train_loader, desc=f"Epoch {epoch + 1}")
):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images)

        loss = criterion(outputs.logits, labels)

        loss.backward()

        optimizer.step()
        if (batch_idx + 1) % 500 == 0:

           torch.save({
              "epoch": epoch,
              "batch": batch_idx + 1,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict()
           }, "checkpoint.pth")

           print(f"Checkpoint saved at batch {batch_idx + 1}")
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1} completed | "
        f"Average Loss: {average_loss:.4f}"
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)

            predictions = outputs.logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"Validation Accuracy: {accuracy:.4f}")
for param in model.vit.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():

    if param.requires_grad:
        print("Trainable:", name)
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_accuracy": accuracy
}, "best_model.pth")

print("Best model saved!")
