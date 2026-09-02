from torch.utils.data import DataLoader, random_split
from dataset import APTOSDataset
import torch

# Load full dataset
full_dataset = APTOSDataset("train.csv", "train_images")
# Split: 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # for reproducibility
)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")

# Wrap in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Quick test: pull one batch
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels)