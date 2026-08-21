from pathlib import Path
from datasets import load_dataset


DATA_DIR = Path("data/raw")


def load_plantvillage():
    """Load the PlantVillage dataset metadata."""
    train = load_dataset(
        "mohanty/PlantVillage",
        "default",
        split="train"
    )

    test = load_dataset(
        "mohanty/PlantVillage",
        "default",
        split="test"
    )

    return train, test


if __name__ == "__main__":
    train, test = load_plantvillage()

    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    print("\nFirst training sample:")
    print(train[0])
