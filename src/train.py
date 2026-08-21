import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import ViTForImageClassification

from dataset import train_loader, val_loader


# -------------------------
# Model
# -------------------------

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=38,
    ignore_mismatched_sizes=True
)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)


# -------------------------
# Optimizer and Loss
# -------------------------

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

criterion = CrossEntropyLoss()


# -------------------------
# Training
# -------------------------

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

        total_loss += loss.item()

        # Print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

        # Save checkpoint every 500 batches
        if (batch_idx + 1) % 500 == 0:

            torch.save({
                "epoch": epoch,
                "batch": batch_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, "checkpoint.pth")

            print(
                f"Checkpoint saved at batch {batch_idx + 1}"
            )


# -------------------------
# Validation
# -------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images)

        predictions = outputs.logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


accuracy = correct / total

print("Validation Accuracy:", accuracy)


# -------------------------
# Save Best Model
# -------------------------

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_accuracy": accuracy
}, "best_model.pth")

print("Best model saved!")