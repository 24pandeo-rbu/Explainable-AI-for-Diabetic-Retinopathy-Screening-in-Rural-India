"""
Retinal Structure Segmentation Module.

OpenCV-based extraction of clinically relevant retinal structures:
- Blood vessel segmentation
- Optic disc localization
- Exudate detection
- Hemorrhage detection
- Microaneurysm candidate detection
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    """Container for all retinal structure segmentation outputs."""
    vessels_mask: np.ndarray         # Binary vessel map
    optic_disc_center: Tuple[int, int]  # (x, y) center of optic disc
    optic_disc_radius: int           # Estimated radius
    exudates_mask: np.ndarray        # Binary exudate map
    hemorrhages_mask: np.ndarray     # Binary hemorrhage map
    microaneurysm_candidates: list   # List of (x, y, radius) tuples
    composite_overlay: np.ndarray    # Colored overlay on original image


def extract_green_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Extract and enhance the green channel — most informative for retinal structures."""
    green = image_bgr[:, :, 1]
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(green)


def create_fundus_mask(image_bgr: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Create a binary mask of the fundus region (exclude dark borders)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment_vessels(image_bgr: np.ndarray, fundus_mask: Optional[np.ndarray] = None
                    ) -> np.ndarray:
    """
    Segment retinal blood vessels using green-channel CLAHE + 
    adaptive thresholding + morphological skeletonization.
    
    Returns binary vessel mask.
    """
    green_enhanced = extract_green_channel(image_bgr)
    
    # Invert — vessels are darker than background
    inverted = cv2.bitwise_not(green_enhanced)
    
    # Morphological top-hat to extract thin structures (vessels)
    kernel_size = max(15, min(image_bgr.shape[:2]) // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel)
    
    # Adaptive threshold
    block_size = max(11, min(image_bgr.shape[:2]) // 20)
    if block_size % 2 == 0:
        block_size += 1
    vessels = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
        block_size, -2
    )
    
    # Clean up with morphological operations
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, small_kernel)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, small_kernel)
    
    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessels, connectivity=8)
    min_area = vessels.shape[0] * vessels.shape[1] * 0.0001
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            vessels[labels == i] = 0
    
    # Apply fundus mask
    if fundus_mask is not None:
        vessels = cv2.bitwise_and(vessels, fundus_mask)
    
    return vessels


def localize_optic_disc(image_bgr: np.ndarray, 
                         fundus_mask: Optional[np.ndarray] = None
                         ) -> Tuple[Tuple[int, int], int]:
    """
    Localize the optic disc as the brightest roughly-circular region.
    
    Returns:
        ((center_x, center_y), estimated_radius)
    """
    # Use the value channel for brightness detection
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # Heavy blur to smooth out small bright spots
    blur_size = max(31, min(image_bgr.shape[:2]) // 10)
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(v_channel, (blur_size, blur_size), 0)
    
    if fundus_mask is not None:
        blurred = cv2.bitwise_and(blurred, fundus_mask)
    
    # Find brightest point
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    
    # Estimate disc radius (typically ~1/7 of image height)
    estimated_radius = max(20, min(image_bgr.shape[:2]) // 14)
    
    return max_loc, estimated_radius


def detect_exudates(image_bgr: np.ndarray, 
                    optic_disc_center: Tuple[int, int],
                    optic_disc_radius: int,
                    fundus_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Detect hard exudates — bright yellowish lesions.
    Uses L*a*b* color space thresholding.
    
    Returns binary exudate mask.
    """
    # Convert to L*a*b* color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    b_channel = lab[:, :, 2]  # Yellow-blue axis
    
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Exudates are bright (high L) and yellowish (high b)
    # Adaptive thresholding based on image statistics
    l_thresh = np.percentile(l_enhanced[l_enhanced > 0], 92)
    b_thresh = np.percentile(b_channel[b_channel > 0], 85)
    
    bright_mask = l_enhanced > l_thresh
    yellow_mask = b_channel > b_thresh
    
    exudates = np.uint8((bright_mask & yellow_mask) * 255)
    
    # Remove optic disc region (it's also bright)
    od_mask = np.zeros_like(exudates)
    cv2.circle(od_mask, optic_disc_center, int(optic_disc_radius * 1.5), 255, -1)
    exudates = cv2.bitwise_and(exudates, cv2.bitwise_not(od_mask))
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    exudates = cv2.morphologyEx(exudates, cv2.MORPH_OPEN, kernel)
    exudates = cv2.morphologyEx(exudates, cv2.MORPH_CLOSE, kernel)
    
    if fundus_mask is not None:
        exudates = cv2.bitwise_and(exudates, fundus_mask)
    
    return exudates


def detect_hemorrhages(image_bgr: np.ndarray, vessels_mask: np.ndarray,
                       fundus_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Detect hemorrhages — dark red lesions in the retina.
    Extracts dark regions in the green channel, excluding vessels.
    
    Returns binary hemorrhage mask.
    """
    green_enhanced = extract_green_channel(image_bgr)
    
    # Hemorrhages are dark regions in the green channel
    # Use morphological bottom-hat to extract dark features
    kernel_size = max(15, min(image_bgr.shape[:2]) // 25)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(green_enhanced, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold
    thresh = max(15, np.percentile(blackhat[blackhat > 0], 90)) if np.any(blackhat > 0) else 15
    _, hemorrhages = cv2.threshold(blackhat, int(thresh), 255, cv2.THRESH_BINARY)
    
    # Remove vessels (they're also dark)
    dilated_vessels = cv2.dilate(vessels_mask, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    hemorrhages = cv2.bitwise_and(hemorrhages, cv2.bitwise_not(dilated_vessels))
    
    # Morphological cleanup — hemorrhages should be blob-like
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hemorrhages = cv2.morphologyEx(hemorrhages, cv2.MORPH_OPEN, kernel)
    hemorrhages = cv2.morphologyEx(hemorrhages, cv2.MORPH_CLOSE, kernel)
    
    # Filter by circularity — hemorrhages tend to be roughly circular
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hemorrhages)
    filtered = np.zeros_like(hemorrhages)
    min_area = 10
    max_area = hemorrhages.shape[0] * hemorrhages.shape[1] * 0.01
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 255
    
    if fundus_mask is not None:
        filtered = cv2.bitwise_and(filtered, fundus_mask)
    
    return filtered


def detect_microaneurysms(image_bgr: np.ndarray, vessels_mask: np.ndarray,
                          fundus_mask: Optional[np.ndarray] = None
                          ) -> list:
    """
    Detect microaneurysm candidates using blob detection on the enhanced green channel.
    
    Returns list of (x, y, radius) tuples.
    """
    green_enhanced = extract_green_channel(image_bgr)
    
    # Invert for blob detection (MAs are small dark dots)
    inverted = cv2.bitwise_not(green_enhanced)
    
    # Remove vessels to avoid false positives
    dilated_vessels = cv2.dilate(vessels_mask,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    inverted = cv2.bitwise_and(inverted, cv2.bitwise_not(dilated_vessels))
    
    if fundus_mask is not None:
        inverted = cv2.bitwise_and(inverted, fundus_mask)
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = max(50, min(image_bgr.shape[:2]) // 10 * 3)
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 50
    params.maxThreshold = 255
    params.thresholdStep = 10
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    
    candidates = [(int(kp.pt[0]), int(kp.pt[1]), int(kp.size / 2)) for kp in keypoints]
    
    return candidates


def create_composite_overlay(image_bgr: np.ndarray,
                              vessels_mask: np.ndarray,
                              exudates_mask: np.ndarray,
                              hemorrhages_mask: np.ndarray,
                              microaneurysms: list,
                              optic_disc_center: Tuple[int, int],
                              optic_disc_radius: int) -> np.ndarray:
    """
    Create a color-coded composite overlay showing all detected structures.
    
    Color coding:
    - Red: Blood vessels
    - Yellow: Exudates
    - Blue: Hemorrhages
    - Cyan circles: Microaneurysm candidates
    - Green circle: Optic disc
    """
    overlay = image_bgr.copy()
    
    # Vessels in red
    vessel_color = np.zeros_like(overlay)
    vessel_color[:, :, 2] = vessels_mask  # Red channel
    overlay = cv2.addWeighted(overlay, 1.0, vessel_color, 0.3, 0)
    
    # Exudates in yellow
    exudate_color = np.zeros_like(overlay)
    exudate_color[:, :, 1] = exudates_mask  # Green
    exudate_color[:, :, 2] = exudates_mask  # Red
    overlay = cv2.addWeighted(overlay, 1.0, exudate_color, 0.4, 0)
    
    # Hemorrhages in blue
    hemorrhage_color = np.zeros_like(overlay)
    hemorrhage_color[:, :, 0] = hemorrhages_mask  # Blue
    overlay = cv2.addWeighted(overlay, 1.0, hemorrhage_color, 0.4, 0)
    
    # Microaneurysms as cyan circles
    for (x, y, r) in microaneurysms:
        cv2.circle(overlay, (x, y), max(r, 3), (255, 255, 0), 2)
        cv2.circle(overlay, (x, y), max(r, 3) + 2, (0, 255, 255), 1)
    
    # Optic disc as green circle
    cv2.circle(overlay, optic_disc_center, optic_disc_radius, (0, 255, 0), 2)
    cv2.circle(overlay, optic_disc_center, optic_disc_radius + 3, (0, 200, 0), 1)
    
    return overlay


def run_full_segmentation(image_bgr: np.ndarray) -> SegmentationResult:
    """
    Run the complete retinal structure segmentation pipeline.
    
    Args:
        image_bgr: Input fundus image in BGR format.
    
    Returns:
        SegmentationResult with all detected structures.
    """
    # Create fundus mask
    fundus_mask = create_fundus_mask(image_bgr)
    
    # Step 1: Vessel segmentation
    vessels = segment_vessels(image_bgr, fundus_mask)
    
    # Step 2: Optic disc localization
    od_center, od_radius = localize_optic_disc(image_bgr, fundus_mask)
    
    # Step 3: Exudate detection
    exudates = detect_exudates(image_bgr, od_center, od_radius, fundus_mask)
    
    # Step 4: Hemorrhage detection
    hemorrhages = detect_hemorrhages(image_bgr, vessels, fundus_mask)
    
    # Step 5: Microaneurysm detection
    microaneurysms = detect_microaneurysms(image_bgr, vessels, fundus_mask)
    
    # Step 6: Create composite overlay
    composite = create_composite_overlay(
        image_bgr, vessels, exudates, hemorrhages, 
        microaneurysms, od_center, od_radius
    )
    
    return SegmentationResult(
        vessels_mask=vessels,
        optic_disc_center=od_center,
        optic_disc_radius=od_radius,
        exudates_mask=exudates,
        hemorrhages_mask=hemorrhages,
        microaneurysm_candidates=microaneurysms,
        composite_overlay=composite,
    )


def get_lesion_counts(result: SegmentationResult) -> Dict:
    """
    Summarize detected lesion counts for the clinical report.
    """
    # Count exudate regions
    n_exudates, _, _, _ = cv2.connectedComponentsWithStats(result.exudates_mask)
    n_exudates = max(0, n_exudates - 1)  # subtract background
    
    # Count hemorrhage regions
    n_hemorrhages, _, _, _ = cv2.connectedComponentsWithStats(result.hemorrhages_mask)
    n_hemorrhages = max(0, n_hemorrhages - 1)
    
    # Vessel density
    fundus_area = np.sum(create_fundus_mask(
        np.zeros((result.vessels_mask.shape[0], result.vessels_mask.shape[1], 3), 
                  dtype=np.uint8)) > 0) or 1
    # Use vessel mask area as a fraction of total mask
    total_pixels = result.vessels_mask.shape[0] * result.vessels_mask.shape[1]
    vessel_density = np.sum(result.vessels_mask > 0) / max(total_pixels, 1)
    
    return {
        "exudate_count": n_exudates,
        "hemorrhage_count": n_hemorrhages,
        "microaneurysm_count": len(result.microaneurysm_candidates),
        "vessel_density_pct": round(vessel_density * 100, 2),
        "optic_disc_detected": result.optic_disc_center is not None,
    }
