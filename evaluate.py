import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataloader import val_loader
from model import build_model

device = torch.device("cpu")

# Load the trained model
model = build_model(num_classes=5).to(device)
model.load_state_dict(torch.load("eye_coniq_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=3))

print("Confusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(all_labels, all_preds))