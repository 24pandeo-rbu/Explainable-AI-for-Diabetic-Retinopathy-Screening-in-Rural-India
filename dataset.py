import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class APTOSDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        label = row["diagnosis"]

        img_path = os.path.join(self.image_folder, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Quick test when run directly
if __name__ == "__main__":
    dataset = APTOSDataset("train.csv", "train_images")
    print("Dataset size:", len(dataset))

    image, label = dataset[0]
    print("Image shape:", image.shape)
    print("Label:", label)