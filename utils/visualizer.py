"""
Real-Time Visualization and HUD Dashboard for Object Tracking Comparison.

Features:
- Dual display modes: Full Combined Overlay vs. Split-Screen (side-by-side)
- Particle cloud rendering with alpha transparency
- Kalman filter 95% confidence covariance error ellipse
- Velocity vector arrows and fading trajectory trails
- Semi-transparent Telemetry HUD with FPS, latencies, error, and occlusion status
- Interactive controls guide banner
"""

from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np

from detectors.detector import DetectionResult
from utils.metrics import FrameRecord


class RenderMode(Enum):
    OVERLAY = "overlay"
    SPLIT_SCREEN = "split_screen"


# BGR Color Palette
COLOR_DETECTION_RED = (0, 0, 255)       # Raw detector
COLOR_KF_BLUE = (255, 140, 0)           # Kalman Filter (Deep Sky Blue / Dodger Blue in BGR: 255, 140, 0 is orange, let's use BGR: B=240, G=120, R=0)
COLOR_KF_BLUE_BGR = (255, 120, 0)       # Cyan-Blue
COLOR_PF_PARTICLE_GREEN = (50, 205, 50) # Particles (Lime Green)
COLOR_PF_MEAN_YELLOW = (0, 215, 255)    # PF Centroid (Yellow/Gold in BGR)
COLOR_OCCLUSION_DARK = (40, 40, 40)     # Occlusion zone
COLOR_HUD_BG = (20, 20, 20)             # HUD Dark Background
COLOR_TEXT_WHITE = (240, 240, 240)
COLOR_STATUS_OK = (80, 220, 80)
COLOR_STATUS_WARN = (0, 69, 255)        # Red-Orange


class TrackingVisualizer:
    """
    Renders tracking overlays, particle clouds, uncertainty ellipses,
    trajectory history, and telemetry HUD.
    """

    def __init__(
        self,
        trail_length: int = 60,
        render_mode: RenderMode = RenderMode.OVERLAY,
        max_rendered_particles: int = 350,
    ) -> None:
        """
        Args:
            trail_length: Max points stored in trajectory trail.
            render_mode: RenderMode.OVERLAY or RenderMode.SPLIT_SCREEN.
            max_rendered_particles: Max particles drawn on screen for high FPS.
        """
        self.trail_length: int = int(trail_length)
        self.render_mode: RenderMode = render_mode
        self.max_rendered_particles: int = int(max_rendered_particles)

        # Trajectory trail histories
        self.raw_trail: Deque[Tuple[int, int]] = deque(maxlen=self.trail_length)
        self.kf_trail: Deque[Tuple[int, int]] = deque(maxlen=self.trail_length)
        self.pf_trail: Deque[Tuple[int, int]] = deque(maxlen=self.trail_length)

    def clear_trails(self) -> None:
        """Clear all historical trajectory trails."""
        self.raw_trail.clear()
        self.kf_trail.clear()
        self.pf_trail.clear()

    def toggle_mode(self) -> RenderMode:
        """Switch between Overlay and Split-screen modes."""
        if self.render_mode == RenderMode.OVERLAY:
            self.render_mode = RenderMode.SPLIT_SCREEN
        else:
            self.render_mode = RenderMode.OVERLAY
        return self.render_mode

    def draw_covariance_ellipse(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        covariance_2d: np.ndarray,
        color: Tuple[int, int, int] = COLOR_KF_BLUE_BGR,
        confidence_scale: float = 2.4477,  # 95% confidence interval for 2 DOF
    ) -> None:
        """
        Draw 2D error covariance ellipse using eigen-decomposition.
        """
        cx, cy = int(round(center[0])), int(round(center[1]))
        if covariance_2d.shape != (2, 2) or np.isnan(covariance_2d).any():
            return

        # Ensure matrix symmetry
        cov_sym = 0.5 * (covariance_2d + covariance_2d.T)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov_sym)
        except np.linalg.LinAlgError:
            return

        # Clamp eigenvalues to prevent negative square roots from numerical noise
        eigenvals = np.maximum(eigenvals, 1e-4)

        # Sort eigenvalues largest first
        order = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Calculate semi-major and semi-minor axis lengths
        axes = (
            max(2, int(round(confidence_scale * np.sqrt(eigenvals[0])))),
            max(2, int(round(confidence_scale * np.sqrt(eigenvals[1])))),
        )

        # Calculate angle of rotation in degrees
        angle_deg = float(np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])))

        # Draw semi-transparent ellipse
        overlay = image.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle_deg, 0, 360, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    def draw_trail(
        self,
        image: np.ndarray,
        trail: Deque[Tuple[int, int]],
        color: Tuple[int, int, int],
        base_thickness: int = 2,
    ) -> None:
        """Draw fading motion path trail with variable thickness."""
        n = len(trail)
        if n < 2:
            return

        for i in range(1, n):
            thickness = max(1, int(round(base_thickness * (i / n))))
            cv2.line(image, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

    def draw_velocity_vector(
        self,
        image: np.ndarray,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        color: Tuple[int, int, int],
        scale: float = 0.5,
    ) -> None:
        """Draw directed velocity vector arrow."""
        x, y = int(round(pos[0])), int(round(pos[1]))
        vx, vy = vel
        end_x = int(round(x + (vx * scale)))
        end_y = int(round(y + (vy * scale)))

        speed = float(np.hypot(vx, vy))
        if speed > 2.0:
            cv2.arrowedLine(image, (x, y), (end_x, end_y), color, 2, cv2.LINE_AA, tipLength=0.25)

    def draw_spatial_occlusion_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> None:
        """Draw semi-transparent occlusion barrier rectangle."""
        sx, sy, sw, sh = box
        overlay = image.copy()
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (30, 30, 30), -1)
        cv2.rectangle(image, (sx, sy), (sx + sw, sy + sh), (80, 80, 80), 2)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Label inside box
        cv2.putText(
            image,
            "OCCLUSION BARRIER",
            (sx + 10, sy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def draw_particles(
        self,
        image: np.ndarray,
        particles_2d: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Draw particle cloud as green dots."""
        if len(particles_2d) == 0:
            return

        h, w = image.shape[:2]
        # Downsample particles if too many for fluid rendering
        if len(particles_2d) > self.max_rendered_particles:
            indices = np.random.choice(len(particles_2d), size=self.max_rendered_particles, replace=False)
            pts = particles_2d[indices]
        else:
            pts = particles_2d

        for p in pts:
            px, py = int(round(p[0])), int(round(p[1]))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 1, COLOR_PF_PARTICLE_GREEN, -1, cv2.LINE_AA)

    def render_hud(
        self,
        image: np.ndarray,
        fps: float,
        record: Optional[FrameRecord],
        is_occluded: bool,
        is_manual_occlusion: bool,
    ) -> None:
        """Render modern telemetry HUD panel."""
        h, w = image.shape[:2]

        # Top banner background: semi-transparent black rectangle
        hud_height = 86
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), COLOR_HUD_BG, -1)
        # Bottom controls banner
        cv2.rectangle(overlay, (0, h - 28), (w, h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

        # Border separator lines
        cv2.line(image, (0, hud_height), (w, hud_height), (60, 60, 60), 1)
        cv2.line(image, (0, h - 28), (w, h - 28), (60, 60, 60), 1)

        # Status text & color
        if is_manual_occlusion or is_occluded:
            status_text = "[OCCLUDED]"
            status_color = COLOR_STATUS_WARN
        else:
            status_text = "[TRACKING]"
            status_color = COLOR_STATUS_OK

        # Line 1: Header + FPS + Status
        line1 = f"REAL-TIME TRACKING HUD | FPS: {fps:4.1f} | STATUS: "
        cv2.putText(image, line1, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        status_offset = int(12 + (len(line1) * 8.8))
        cv2.putText(image, status_text, (status_offset, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        # Mode tag on top right
        mode_str = f"MODE: {self.render_mode.value.upper()}"
        cv2.putText(image, mode_str, (w - 190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        if record is not None:
            # Line 2: Kalman Filter Telemetry (Blue)
            kf_text = (
                f"KF (Blue): Latency: {record.kf_latency_ms:5.2f}ms | "
                f"Error: {record.kf_error:5.1f}px | "
                f"Vel: ({record.kf_vel[0]:+5.1f}, {record.kf_vel[1]:+5.1f})"
            )
            cv2.putText(image, kf_text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_KF_BLUE_BGR, 1, cv2.LINE_AA)

            # Line 3: Particle Filter Telemetry (Yellow)
            pf_text = (
                f"PF (Yellow): Latency: {record.pf_latency_ms:5.2f}ms | "
                f"Error: {record.pf_error:5.1f}px | "
                f"N_eff: {record.pf_n_eff:5.1f} | "
                f"Vel: ({record.pf_vel[0]:+5.1f}, {record.pf_vel[1]:+5.1f})"
            )
            cv2.putText(image, pf_text, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_PF_MEAN_YELLOW, 1, cv2.LINE_AA)

        # Bottom shortcut guide
        guide = "[Q]uit  |  [S]elect ROI  |  [O]cclusion Toggle  |  [P]ause  |  [C]lear Trails  |  [M]ode Toggle"
        cv2.putText(image, guide, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def render_frame(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        kf_state: np.ndarray,
        kf_cov: Optional[np.ndarray],
        pf_state: np.ndarray,
        pf_particles: Tuple[np.ndarray, np.ndarray],
        fps: float,
        record: Optional[FrameRecord] = None,
        is_manual_occlusion: bool = False,
        spatial_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Assemble final rendered visualization frame according to active render mode.
        """
        kf_pos = (float(kf_state[0]), float(kf_state[1]))
        kf_vel = (float(kf_state[2]), float(kf_state[3]))
        pf_pos = (float(pf_state[0]), float(pf_state[1]))
        pf_vel = (float(pf_state[2]), float(pf_state[3]))

        # Update trajectory queues
        if detection.centroid is not None:
            self.raw_trail.append((int(round(detection.centroid[0])), int(round(detection.centroid[1]))))
        self.kf_trail.append((int(round(kf_pos[0])), int(round(kf_pos[1]))))
        self.pf_trail.append((int(round(pf_pos[0])), int(round(pf_pos[1]))))

        if self.render_mode == RenderMode.OVERLAY:
            canvas = frame.copy()

            # 1. Spatial occlusion box if present
            if spatial_box is not None:
                self.draw_spatial_occlusion_box(canvas, spatial_box)

            # 2. Draw raw detection bounding box & centroid
            if detection.is_detected and not detection.is_occluded and detection.centroid is not None:
                if detection.bbox is not None:
                    bx, by, bw, bh = detection.bbox
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_DETECTION_RED, 2)
                cx, cy = int(round(detection.centroid[0])), int(round(detection.centroid[1]))
                cv2.circle(canvas, (cx, cy), 5, COLOR_DETECTION_RED, -1, cv2.LINE_AA)

            # 3. Particle cloud
            pts_2d, w = pf_particles
            self.draw_particles(canvas, pts_2d, w)

            # 4. Motion trails
            self.draw_trail(canvas, self.raw_trail, COLOR_DETECTION_RED, 1)
            self.draw_trail(canvas, self.kf_trail, COLOR_KF_BLUE_BGR, 2)
            self.draw_trail(canvas, self.pf_trail, COLOR_PF_MEAN_YELLOW, 2)

            # 5. Covariance ellipse for Kalman filter
            if kf_cov is not None:
                self.draw_covariance_ellipse(canvas, kf_pos, kf_cov[:2, :2], COLOR_KF_BLUE_BGR)

            # 6. Current estimated positions & velocity vectors
            # KF estimate marker (Blue circle)
            kfx, kfy = int(round(kf_pos[0])), int(round(kf_pos[1]))
            cv2.circle(canvas, (kfx, kfy), 7, COLOR_KF_BLUE_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (kfx, kfy), 9, (255, 255, 255), 1, cv2.LINE_AA)
            self.draw_velocity_vector(canvas, kf_pos, kf_vel, COLOR_KF_BLUE_BGR)

            # PF estimate marker (Yellow ring with green core)
            pfx, pfy = int(round(pf_pos[0])), int(round(pf_pos[1]))
            cv2.circle(canvas, (pfx, pfy), 7, COLOR_PF_MEAN_YELLOW, -1, cv2.LINE_AA)
            cv2.circle(canvas, (pfx, pfy), 9, (0, 0, 0), 1, cv2.LINE_AA)
            self.draw_velocity_vector(canvas, pf_pos, pf_vel, COLOR_PF_MEAN_YELLOW)

            # 7. Render Telemetry HUD
            self.render_hud(canvas, fps, record, detection.is_occluded, is_manual_occlusion)
            return canvas

        else:
            # SPLIT_SCREEN mode
            left = frame.copy()
            right = frame.copy()

            if spatial_box is not None:
                self.draw_spatial_occlusion_box(left, spatial_box)
                self.draw_spatial_occlusion_box(right, spatial_box)

            # Draw raw detection on both
            if detection.is_detected and not detection.is_occluded and detection.centroid is not None:
                cx, cy = int(round(detection.centroid[0])), int(round(detection.centroid[1]))
                if detection.bbox is not None:
                    bx, by, bw, bh = detection.bbox
                    cv2.rectangle(left, (bx, by), (bx + bw, by + bh), COLOR_DETECTION_RED, 2)
                    cv2.rectangle(right, (bx, by), (bx + bw, by + bh), COLOR_DETECTION_RED, 2)
                cv2.circle(left, (cx, cy), 5, COLOR_DETECTION_RED, -1)
                cv2.circle(right, (cx, cy), 5, COLOR_DETECTION_RED, -1)

            # Left Panel: Kalman Filter only
            self.draw_trail(left, self.kf_trail, COLOR_KF_BLUE_BGR, 2)
            if kf_cov is not None:
                self.draw_covariance_ellipse(left, kf_pos, kf_cov[:2, :2], COLOR_KF_BLUE_BGR)
            kfx, kfy = int(round(kf_pos[0])), int(round(kf_pos[1]))
            cv2.circle(left, (kfx, kfy), 7, COLOR_KF_BLUE_BGR, -1)
            self.draw_velocity_vector(left, kf_pos, kf_vel, COLOR_KF_BLUE_BGR)
            cv2.putText(left, "KALMAN FILTER", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_KF_BLUE_BGR, 2)

            # Right Panel: Particle Filter only
            pts_2d, w = pf_particles
            self.draw_particles(right, pts_2d, w)
            self.draw_trail(right, self.pf_trail, COLOR_PF_MEAN_YELLOW, 2)
            pfx, pfy = int(round(pf_pos[0])), int(round(pf_pos[1]))
            cv2.circle(right, (pfx, pfy), 7, COLOR_PF_MEAN_YELLOW, -1)
            self.draw_velocity_vector(right, pf_pos, pf_vel, COLOR_PF_MEAN_YELLOW)
            cv2.putText(right, "PARTICLE FILTER", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PF_MEAN_YELLOW, 2)

            # Concatenate horizontally
            combined = np.hstack([left, right])
            # Center dividing line
            mid_x = left.shape[1]
            cv2.line(combined, (mid_x, 0), (mid_x, combined.shape[0]), (100, 100, 100), 2)

            self.render_hud(combined, fps, record, detection.is_occluded, is_manual_occlusion)
            return combined
