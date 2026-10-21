"""
Color / HSV-Based Blob Detector with Synthetic Noise and Occlusion Injection.

Extracts 2D object centroid measurements from video frames using color segmentation,
contour moments, and morphological operations.
Supports interactive ROI calibration, synthetic measurement noise injection,
periodic and spatial simulated occlusions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Standardized output container for object detector."""
    centroid: Optional[Tuple[float, float]]       # Measurement passed to trackers (with noise/occlusion)
    raw_centroid: Optional[Tuple[float, float]]   # Uncorrupted measurement (ground-truth reference)
    bbox: Optional[Tuple[int, int, int, int]]     # (x, y, w, h)
    contour: Optional[np.ndarray]                 # Raw contour points
    mask: Optional[np.ndarray]                    # Binary segmentation mask
    is_detected: bool                             # True if object visually detected
    is_occluded: bool                             # True if measurement is currently suppressed
    confidence: float                             # Detection confidence score (0.0 to 1.0)


# Built-in HSV color range presets [Lower_HSV, Upper_HSV]
COLOR_PRESETS: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "orange": [
        (np.array([5, 120, 120], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
    ],
    "green": [
        (np.array([35, 80, 80], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))
    ],
    "blue": [
        (np.array([90, 100, 100], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8))
    ],
    "yellow": [
        (np.array([20, 100, 100], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
    ],
    "cyan": [
        (np.array([80, 100, 100], dtype=np.uint8), np.array([100, 255, 255], dtype=np.uint8))
    ],
    # Red spans across Hue 0/180 boundary
    "red": [
        (np.array([0, 120, 100], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        (np.array([170, 120, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    ],
}


class ColorBlobDetector:
    """
    HSV Color Segmentation Detector with measurement corruption simulators.
    """

    def __init__(
        self,
        color_preset: str = "orange",
        min_area: float = 100.0,
        max_area: float = 200000.0,
        measurement_noise_std: float = 0.0,
        periodic_occlusion_interval: int = 0,
        periodic_occlusion_duration: int = 0,
        spatial_occlusion_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Args:
            color_preset: Name of preset color ('orange', 'green', 'blue', 'red', 'yellow').
            min_area: Minimum contour area threshold in pixels.
            max_area: Maximum contour area threshold in pixels.
            measurement_noise_std: Synthetic Gaussian measurement noise std in pixels.
            periodic_occlusion_interval: Frame period for synthetic occlusion (0 = disabled).
            periodic_occlusion_duration: Number of frames to occlude during each period.
            spatial_occlusion_box: (x, y, w, h) zone where target is suppressed if within bounds.
        """
        self.color_preset: str = color_preset
        self.min_area: float = float(min_area)
        self.max_area: float = float(max_area)
        self.measurement_noise_std: float = float(measurement_noise_std)
        self.periodic_occlusion_interval: int = int(periodic_occlusion_interval)
        self.periodic_occlusion_duration: int = int(periodic_occlusion_duration)
        self.spatial_occlusion_box: Optional[Tuple[int, int, int, int]] = spatial_occlusion_box

        # Manual occlusion toggle
        self.is_manual_occlusion: bool = False

        # Internal frame counter
        self.frame_index: int = 0

        # Active HSV ranges
        self.hsv_ranges: List[Tuple[np.ndarray, np.ndarray]] = COLOR_PRESETS.get(
            color_preset.lower(), COLOR_PRESETS["orange"]
        )

        # Morphological kernels
        self.morph_kernel_open: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.morph_kernel_close: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def set_color_ranges(self, ranges: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Explicitly override HSV bounds."""
        self.hsv_ranges = ranges

    def calibrate_from_roi(
        self,
        frame: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        h_tol: int = 15,
        s_tol: int = 60,
        v_tol: int = 60,
    ) -> None:
        """
        Calibrate HSV detection thresholds automatically from a selected bounding box.

        Args:
            frame: Input BGR frame.
            roi_bbox: (x, y, w, h) selected region of interest.
            h_tol: Hue tolerance margin.
            s_tol: Saturation tolerance margin.
            v_tol: Value tolerance margin.
        """
        x, y, w, h = roi_bbox
        x, y = max(0, x), max(0, y)
        roi = frame[y : y + h, x : x + w]

        if roi.size == 0:
            return

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel = hsv_roi[:, :, 0]
        s_channel = hsv_roi[:, :, 1]
        v_channel = hsv_roi[:, :, 2]

        # Calculate median color values to resist background outliers
        h_median = int(np.median(h_channel))
        s_median = int(np.median(s_channel))
        v_median = int(np.median(v_channel))

        lower_h = max(0, h_median - h_tol)
        upper_h = min(179, h_median + h_tol)
        lower_s = max(40, s_median - s_tol)
        upper_s = min(255, s_median + s_tol)
        lower_v = max(40, v_median - v_tol)
        upper_v = min(255, v_median + v_tol)

        self.hsv_ranges = [
            (np.array([lower_h, lower_s, lower_v], dtype=np.uint8),
             np.array([upper_h, upper_s, upper_v], dtype=np.uint8))
        ]

    def toggle_manual_occlusion(self) -> bool:
        """Toggle manual occlusion state and return new state."""
        self.is_manual_occlusion = not self.is_manual_occlusion
        return self.is_manual_occlusion

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect object centroid in frame and apply synthetic corruption.

        Args:
            frame: BGR input image.

        Returns:
            DetectionResult container.
        """
        self.frame_index += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute composite binary mask
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower_b, upper_b in self.hsv_ranges:
            sub_mask = cv2.inRange(hsv, lower_b, upper_b)
            combined_mask = cv2.bitwise_or(combined_mask, sub_mask)

        # Morphological noise removal
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel_open)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self.morph_kernel_close)

        # Contour extraction
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_centroid: Optional[Tuple[float, float]] = None
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_contour: Optional[np.ndarray] = None
        confidence: float = 0.0

        if contours:
            # Pick largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if self.min_area <= area <= self.max_area:
                moments = cv2.moments(largest_contour)
                if moments["m00"] > 1e-4:
                    cx = float(moments["m10"] / moments["m00"])
                    cy = float(moments["m01"] / moments["m00"])
                    raw_centroid = (cx, cy)
                    best_bbox = cv2.boundingRect(largest_contour)
                    best_contour = largest_contour
                    # Simple confidence based on circularity / compactness
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter ** 2))
                        confidence = float(np.clip(circularity, 0.1, 1.0))
                    else:
                        confidence = 0.5

        is_detected = raw_centroid is not None

        # Determine occlusion condition
        is_occluded = False

        if self.is_manual_occlusion:
            is_occluded = True

        if self.periodic_occlusion_interval > 0 and self.periodic_occlusion_duration > 0:
            cycle = self.frame_index % self.periodic_occlusion_interval
            if cycle < self.periodic_occlusion_duration:
                is_occluded = True

        if raw_centroid is not None and self.spatial_occlusion_box is not None:
            sx, sy, sw, sh = self.spatial_occlusion_box
            cx, cy = raw_centroid
            if sx <= cx <= (sx + sw) and sy <= cy <= (sy + sh):
                is_occluded = True

        # Compute output measurement
        measurement: Optional[Tuple[float, float]] = None
        if is_detected and not is_occluded:
            cx, cy = raw_centroid  # type: ignore
            if self.measurement_noise_std > 0.0:
                noise_x = float(np.random.normal(0.0, self.measurement_noise_std))
                noise_y = float(np.random.normal(0.0, self.measurement_noise_std))
                measurement = (cx + noise_x, cy + noise_y)
            else:
                measurement = (cx, cy)

        return DetectionResult(
            centroid=measurement,
            raw_centroid=raw_centroid,
            bbox=best_bbox,
            contour=best_contour,
            mask=cleaned_mask,
            is_detected=is_detected,
            is_occluded=is_occluded,
            confidence=confidence,
        )
