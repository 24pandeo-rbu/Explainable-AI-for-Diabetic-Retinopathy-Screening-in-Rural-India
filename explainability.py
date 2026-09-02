"""
Explainability Module — Grad-CAM, Guided Backpropagation, and Clinical Evidence Mapping.

Generates visual explanations for DR predictions, enabling
ophthalmologist validation in a human-in-the-loop workflow.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional


# Clinical criteria mapped to DR severity levels
DR_CLINICAL_CRITERIA = {
    0: {
        "name": "No DR",
        "findings": "No visible retinal abnormalities",
        "recommendation": "Routine screening in 12 months.",
        "urgency": "none",
        "color": (46, 204, 113),  # green
    },
    1: {
        "name": "Mild NPDR",
        "findings": "Microaneurysms only",
        "recommendation": "Repeat screening in 9-12 months. No immediate referral needed.",
        "urgency": "low",
        "color": (241, 196, 15),  # yellow
    },
    2: {
        "name": "Moderate NPDR",
        "findings": "Microaneurysms, retinal hemorrhages, hard exudates, cotton wool spots",
        "recommendation": "Refer to ophthalmologist within 3-6 months. Monitor for progression.",
        "urgency": "medium",
        "color": (243, 156, 18),  # orange
    },
    3: {
        "name": "Severe NPDR",
        "findings": "Extensive hemorrhages (≥20 in each quadrant), venous beading, IRMA",
        "recommendation": "Urgent referral to retina specialist within 2-4 weeks.",
        "urgency": "high",
        "color": (231, 76, 60),  # red
    },
    4: {
        "name": "Proliferative DR",
        "findings": "Neovascularization, vitreous/preretinal hemorrhage, tractional retinal detachment risk",
        "recommendation": "URGENT: Immediate referral for laser photocoagulation or anti-VEGF therapy.",
        "urgency": "critical",
        "color": (192, 57, 43),  # dark red
    },
}


class GradCAM:
    """
    Grad-CAM implementation for ResNet-18 based DR classification.
    
    Generates class-discriminative localization maps that highlight
    the regions the model considers important for its prediction.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4"):
        """
        Args:
            model: Trained ResNet model.
            target_layer: Name of the convolutional layer to extract activations from.
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks on the target layer
        target = dict(model.named_modules())[target_layer]
        target.register_forward_hook(self._forward_hook)
        target.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, 
                 target_class: Optional[int] = None) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        Generate a Grad-CAM heatmap for the given input.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W].
            target_class: Class index to generate CAM for. If None, uses predicted class.
        
        Returns:
            Tuple of (heatmap as HxW float array [0,1], predicted_class, probabilities).
        """
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM computation
        # Global average pooling of gradients → channel importance weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU — only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class, probs[0].detach()


def create_heatmap_overlay(original_image: np.ndarray, cam: np.ndarray,
                           alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET
                           ) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.
    
    Args:
        original_image: Original BGR image.
        cam: Grad-CAM activation map (any size, will be resized).
        alpha: Blending factor (0 = only image, 1 = only heatmap).
        colormap: OpenCV colormap to use.
    
    Returns:
        BGR image with heatmap overlay.
    """
    h, w = original_image.shape[:2]
    
    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    
    # Blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def create_activation_map(cam: np.ndarray, image_size: Tuple[int, int],
                          threshold: float = 0.5) -> np.ndarray:
    """
    Create a binary activation map highlighting high-attention regions.
    
    Args:
        cam: Grad-CAM activation map.
        image_size: Target (width, height).
        threshold: Activation threshold [0, 1].
    
    Returns:
        Binary mask of high-attention regions.
    """
    cam_resized = cv2.resize(cam.astype(np.float32), image_size)
    _, binary = cv2.threshold(
        np.uint8(255 * cam_resized), int(255 * threshold), 255, cv2.THRESH_BINARY
    )
    return binary


def annotate_with_evidence(original_image: np.ndarray, cam: np.ndarray,
                           predicted_class: int, confidence: float
                           ) -> np.ndarray:
    """
    Create an annotated image with Grad-CAM overlay and clinical evidence markers.
    
    Args:
        original_image: Original BGR image.
        cam: Grad-CAM activation map.
        predicted_class: Predicted DR severity level (0-4).
        confidence: Prediction confidence (0-1).
    
    Returns:
        Annotated BGR image.
    """
    h, w = original_image.shape[:2]
    
    # Create overlay
    overlay = create_heatmap_overlay(original_image, cam, alpha=0.4)
    
    # Get clinical info
    criteria = DR_CLINICAL_CRITERIA[predicted_class]
    color_bgr = criteria["color"][::-1]  # RGB to BGR
    
    # Draw border indicating severity
    border_thickness = max(3, min(h, w) // 80)
    cv2.rectangle(overlay, (0, 0), (w-1, h-1), color_bgr, border_thickness)
    
    # Find high-activation regions and mark them
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    high_activation = cam_resized > 0.6
    
    # Find contours of high activation regions
    mask = np.uint8(high_activation * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles around significant regions
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (h * w * 0.001):  # filter tiny regions
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(overlay, (int(cx), int(cy)), int(radius + 5), color_bgr, 2)
    
    return overlay


def generate_clinical_evidence(cam: np.ndarray, predicted_class: int,
                               probabilities: torch.Tensor
                               ) -> Dict:
    """
    Generate structured clinical evidence from the Grad-CAM analysis.
    
    Returns:
        Dictionary with clinical evidence summary.
    """
    criteria = DR_CLINICAL_CRITERIA[predicted_class]
    
    # Analyze activation pattern
    activation_coverage = np.mean(cam > 0.3)
    peak_activation = cam.max()
    
    # Spatial distribution analysis
    h, w = cam.shape
    quadrants = {
        "superior_temporal": cam[:h//2, :w//2].mean(),
        "superior_nasal": cam[:h//2, w//2:].mean(),
        "inferior_temporal": cam[h//2:, :w//2].mean(),
        "inferior_nasal": cam[h//2:, w//2:].mean(),
    }
    
    most_affected = max(quadrants, key=quadrants.get)
    
    # Compute confidence metrics
    probs = probabilities.cpu().numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    uncertainty = entropy / max_entropy  # Normalized entropy [0,1]
    
    # Referable DR check (Level 2+)
    referable_prob = float(sum(probs[2:]))
    
    evidence = {
        "severity": criteria["name"],
        "severity_level": predicted_class,
        "confidence": float(probs[predicted_class]),
        "uncertainty": float(uncertainty),
        "clinical_findings": criteria["findings"],
        "recommendation": criteria["recommendation"],
        "urgency": criteria["urgency"],
        "activation_coverage": float(activation_coverage),
        "peak_activation": float(peak_activation),
        "most_affected_quadrant": most_affected.replace("_", " ").title(),
        "quadrant_activations": {k.replace("_", " ").title(): float(v) 
                                  for k, v in quadrants.items()},
        "referable_dr_probability": referable_prob,
        "is_referable": referable_prob >= 0.5,
        "class_probabilities": {
            DR_CLINICAL_CRITERIA[i]["name"]: float(probs[i]) 
            for i in range(len(probs))
        },
    }
    
    return evidence


class TemperatureScaling:
    """
    Post-hoc temperature scaling for calibrated confidence scores.
    
    Uses a single temperature parameter learned on a validation set
    to calibrate the model's softmax probabilities.
    """
    
    def __init__(self, temperature: float = 1.5):
        """
        Args:
            temperature: Scaling temperature. Values > 1 soften probabilities
                        (increase uncertainty), values < 1 sharpen them.
        """
        self.temperature = temperature
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to raw logits.
        
        Args:
            logits: Raw model output logits [batch, num_classes].
        
        Returns:
            Calibrated probability distribution.
        """
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)
    
    def optimal_temperature(self, logits_list: List[torch.Tensor], 
                           labels_list: List[int]) -> float:
        """
        Find optimal temperature using NLL minimization on validation data.
        Simple grid search approach.
        """
        best_temp = 1.0
        best_nll = float('inf')
        
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.tensor(labels_list, dtype=torch.long)
        
        for temp in np.arange(0.5, 5.0, 0.1):
            scaled = all_logits / temp
            nll = F.cross_entropy(scaled, all_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
