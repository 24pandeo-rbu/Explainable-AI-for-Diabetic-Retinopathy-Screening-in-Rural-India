"""
Image Quality Assessment & Enhancement Module for Fundus Images.

Evaluates fundus images for adequacy (focus, illumination, field of view)
and applies adaptive enhancement for borderline images.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ImageGrade(Enum):
    """Overall gradeability of a fundus image."""
    ACCEPT = "accept"
    BORDERLINE = "borderline"
    REJECT = "reject"


@dataclass
class QualityReport:
    """Complete quality assessment report for a fundus image."""
    focus_score: float          # 0-100, higher = sharper
    illumination_score: float   # 0-100, higher = better illumination
    fov_score: float            # 0-100, higher = better field of view
    contrast_score: float       # 0-100, higher = better contrast
    overall_score: float        # 0-100, weighted composite
    grade: ImageGrade
    feedback: str               # Human-readable feedback for recapture
    details: dict               # Detailed per-metric breakdown


def assess_focus(image_bgr: np.ndarray) -> Tuple[float, str]:
    """
    Evaluate image sharpness using Laplacian variance.
    Higher variance = sharper image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance — classic blur detection metric
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Also compute Tenengrad (Sobel gradient magnitude)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(gx**2 + gy**2)
    
    # Normalize scores (empirically tuned for fundus images)
    # Typical good fundus: variance > 100, tenengrad > 500
    lap_score = min(100, (variance / 200.0) * 100)
    ten_score = min(100, (tenengrad / 1000.0) * 100)
    
    score = 0.6 * lap_score + 0.4 * ten_score
    
    if score < 25:
        feedback = "Image is very blurry. Please stabilize the camera and refocus."
    elif score < 50:
        feedback = "Image is slightly out of focus. Try recapturing with steady hands."
    else:
        feedback = "Focus quality is acceptable."
    
    return min(100, max(0, score)), feedback


def assess_illumination(image_bgr: np.ndarray) -> Tuple[float, str]:
    """
    Evaluate illumination quality using HSV value channel analysis.
    Checks for over/under-exposure and uniformity.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(float)
    
    mean_brightness = np.mean(v_channel)
    std_brightness = np.std(v_channel)
    
    # Check for over-exposure (too many saturated pixels)
    overexposed_ratio = np.mean(v_channel > 240)
    # Check for under-exposure
    underexposed_ratio = np.mean(v_channel < 15)
    
    # Ideal brightness range for fundus: 80-180
    if mean_brightness < 30:
        brightness_score = (mean_brightness / 30.0) * 40
        feedback = "Image is too dark. Increase flash intensity or ambient lighting."
    elif mean_brightness > 220:
        brightness_score = max(0, (255 - mean_brightness) / 35.0 * 40)
        feedback = "Image is overexposed. Reduce flash intensity."
    elif mean_brightness < 60:
        brightness_score = 40 + ((mean_brightness - 30) / 30.0) * 30
        feedback = "Image is somewhat dark. Consider increasing illumination."
    elif mean_brightness > 200:
        brightness_score = 40 + ((220 - mean_brightness) / 20.0) * 30
        feedback = "Image is somewhat bright. Consider reducing illumination."
    else:
        brightness_score = 70 + min(30, 30 * (1 - abs(mean_brightness - 130) / 70.0))
        feedback = "Illumination is adequate."
    
    # Penalize for extreme exposure
    exposure_penalty = (overexposed_ratio + underexposed_ratio) * 50
    
    # Penalize for non-uniform illumination
    uniformity_penalty = max(0, (std_brightness - 60) / 60.0 * 20)
    
    score = max(0, min(100, brightness_score - exposure_penalty - uniformity_penalty))
    
    return score, feedback


def assess_field_of_view(image_bgr: np.ndarray) -> Tuple[float, str]:
    """
    Check that the fundus image shows adequate retinal field of view.
    Looks for the characteristic circular fundus region.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Threshold to find the bright fundus region vs dark border
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 20.0, "Cannot detect retinal field. Image may be completely dark or obstructed."
    
    # Find the largest contour (should be the fundus circle)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    image_area = h * w
    
    # Calculate what fraction of the image the fundus occupies
    coverage = area / image_area
    
    # Check circularity — fundus should be roughly circular
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    # Score based on coverage (ideal: 50-85% of image area)
    if coverage < 0.15:
        coverage_score = coverage / 0.15 * 30
        feedback = "Very small retinal field. Move camera closer to the eye."
    elif coverage < 0.35:
        coverage_score = 30 + (coverage - 0.15) / 0.20 * 30
        feedback = "Retinal field is small. Adjust camera distance."
    elif coverage > 0.95:
        coverage_score = 60
        feedback = "Image may be cropped. Ensure full fundus is visible."
    else:
        coverage_score = 60 + min(40, (coverage - 0.35) / 0.50 * 40)
        feedback = "Adequate field of view."
    
    # Bonus for circular shape
    circularity_bonus = circularity * 15
    
    score = min(100, max(0, coverage_score + circularity_bonus))
    
    return score, feedback


def assess_contrast(image_bgr: np.ndarray) -> Tuple[float, str]:
    """
    Evaluate image contrast using the green channel (most informative for fundus).
    """
    green = image_bgr[:, :, 1].astype(float)
    
    # Local contrast using standard deviation in patches
    contrast = np.std(green)
    
    # Dynamic range
    p5, p95 = np.percentile(green, [5, 95])
    dynamic_range = p95 - p5
    
    # Score
    contrast_score = min(100, (contrast / 50.0) * 50 + (dynamic_range / 200.0) * 50)
    
    if contrast_score < 30:
        feedback = "Very low contrast. Image may appear washed out."
    elif contrast_score < 50:
        feedback = "Low contrast. Enhancement will be applied."
    else:
        feedback = "Contrast is acceptable."
    
    return min(100, max(0, contrast_score)), feedback


def compute_quality_report(image_bgr: np.ndarray) -> QualityReport:
    """
    Compute a comprehensive image quality report.
    
    Args:
        image_bgr: Input image in BGR format (OpenCV convention).
    
    Returns:
        QualityReport with all metrics and overall grade.
    """
    focus_score, focus_feedback = assess_focus(image_bgr)
    illum_score, illum_feedback = assess_illumination(image_bgr)
    fov_score, fov_feedback = assess_field_of_view(image_bgr)
    contrast_score, contrast_feedback = assess_contrast(image_bgr)
    
    # Weighted composite score
    overall = (
        0.30 * focus_score +
        0.25 * illum_score +
        0.25 * fov_score +
        0.20 * contrast_score
    )
    
    # Determine grade
    if overall >= 55:
        grade = ImageGrade.ACCEPT
        feedback = "✅ Image quality is acceptable for DR screening."
    elif overall >= 35:
        grade = ImageGrade.BORDERLINE
        # Find worst metric
        worst = min(
            [("Focus", focus_score), ("Illumination", illum_score),
             ("Field of View", fov_score), ("Contrast", contrast_score)],
            key=lambda x: x[1]
        )
        feedback = (
            f"⚠️ Borderline quality. Weakest area: {worst[0]} ({worst[1]:.0f}/100). "
            f"Enhancement will be applied automatically."
        )
    else:
        grade = ImageGrade.REJECT
        issues = []
        if focus_score < 35:
            issues.append(focus_feedback)
        if illum_score < 35:
            issues.append(illum_feedback)
        if fov_score < 35:
            issues.append(fov_feedback)
        if contrast_score < 35:
            issues.append(contrast_feedback)
        feedback = "❌ Image rejected. " + " | ".join(issues) if issues else "❌ Image quality too low for reliable screening."
    
    details = {
        "focus": {"score": focus_score, "feedback": focus_feedback},
        "illumination": {"score": illum_score, "feedback": illum_feedback},
        "field_of_view": {"score": fov_score, "feedback": fov_feedback},
        "contrast": {"score": contrast_score, "feedback": contrast_feedback},
    }
    
    return QualityReport(
        focus_score=focus_score,
        illumination_score=illum_score,
        fov_score=fov_score,
        contrast_score=contrast_score,
        overall_score=overall,
        grade=grade,
        feedback=feedback,
        details=details,
    )


# ──────────────────────────────────────────────
# Image Enhancement Pipeline
# ──────────────────────────────────────────────

def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, 
                tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the luminance channel.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def normalize_illumination(image_bgr: np.ndarray, kernel_size: int = 65) -> np.ndarray:
    """
    Normalize illumination using morphological top-hat and bottom-hat transforms.
    Removes uneven illumination patterns common in portable fundus cameras.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Top-hat: bright features on dark background
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # Bottom-hat: dark features on bright background
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine: original + tophat - blackhat normalizes illumination
    corrected = cv2.add(gray, tophat)
    corrected = cv2.subtract(corrected, blackhat)
    
    # Apply correction to all channels proportionally
    result = image_bgr.copy().astype(np.float32)
    gray_float = gray.astype(np.float32)
    gray_float[gray_float == 0] = 1  # avoid division by zero
    
    ratio = corrected.astype(np.float32) / gray_float
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] * ratio, 0, 255)
    
    return result.astype(np.uint8)


def denoise_image(image_bgr: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    Apply non-local means denoising, preserving edges.
    """
    return cv2.fastNlMeansDenoisingColored(
        image_bgr, None, h=strength, hForColorComponents=strength,
        templateWindowSize=7, searchWindowSize=21
    )


def enhance_fundus_image(image_bgr: np.ndarray, 
                         quality_report: QualityReport = None) -> np.ndarray:
    """
    Adaptive enhancement pipeline. Applies different enhancements
    based on the quality report.
    """
    enhanced = image_bgr.copy()
    
    if quality_report is None:
        quality_report = compute_quality_report(image_bgr)
    
    # Always apply mild CLAHE for better feature visibility
    clip = 2.0
    if quality_report.contrast_score < 40:
        clip = 3.5  # Stronger CLAHE for low-contrast images
    elif quality_report.contrast_score < 60:
        clip = 2.5
    
    enhanced = apply_clahe(enhanced, clip_limit=clip)
    
    # Illumination normalization for poorly lit images
    if quality_report.illumination_score < 60:
        enhanced = normalize_illumination(enhanced)
    
    # Denoising for noisy/low-light images
    if quality_report.focus_score < 50 or quality_report.illumination_score < 40:
        enhanced = denoise_image(enhanced, strength=5)
    
    return enhanced
