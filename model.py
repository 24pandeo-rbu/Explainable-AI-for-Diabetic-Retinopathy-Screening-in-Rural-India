import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=5):
    # Load a ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace the final layer to output 5 classes instead of 1000
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def load_trained_model(weights_path="eye_coniq_model.pth", device=None):
    """
    Load a trained model with weights, ready for inference.
    
    Args:
        weights_path: Path to saved model weights.
        device: torch.device to load onto. Defaults to CPU.
    
    Returns:
        Model in eval mode on the specified device.
    """
    if device is None:
        device = torch.device("cpu")
    
    model = build_model(num_classes=5).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def get_gradcam_target_layer(model):
    """
    Get the target layer for Grad-CAM extraction.
    For ResNet-18, this is 'layer4' (the last conv block).
    
    Returns:
        The target layer module.
    """
    return model.layer4


if __name__ == "__main__":
    model = build_model()
    print(model.fc)  # confirm the final layer

    # Quick sanity check: pass a dummy batch through
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)