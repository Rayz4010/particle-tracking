"""
Synthetic Video and Trajectory Generator for Ground-Truth Object Tracking Benchmarking.

Generates realistic video streams and mathematical ground-truth trajectories:
- 2D kinematics (constant velocity, harmonic oscillation, non-linear curves, border collisions)
- Ground truth export with exact position (x, y) and velocity (vx, vy)
- Configurable synthetic occlusion barriers
- Real-time frame generator for live streaming without disk I/O
- MP4 video compilation for reproducible benchmarks
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import csv
import cv2
import numpy as np


class SyntheticStream:
    """
    Real-time synthetic video stream generator yielding (frame, gt_position, gt_velocity, is_occluded).
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        fps: float = 30.0,
        trajectory: str = "mixed",
        ball_radius: int = 22,
        ball_color_bgr: Tuple[int, int, int] = (0, 140, 255),  # Vibrant Orange
        add_occlusion_barrier: bool = True,
        barrier_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Args:
            width: Video frame width in pixels.
            height: Video frame height in pixels.
            fps: Frame rate.
            trajectory: 'linear', 'circular', 'sine', or 'mixed'.
            ball_radius: Radius of the moving ball.
            ball_color_bgr: Ball BGR color (default orange).
            add_occlusion_barrier: Whether to place an occlusion wall.
            barrier_rect: (x, y, w, h) of occlusion obstacle.
        """
        self.width: int = width
        self.height: int = height
        self.fps: float = fps
        self.dt: float = 1.0 / fps
        self.trajectory: str = trajectory
        self.ball_radius: int = ball_radius
        self.ball_color: Tuple[int, int, int] = ball_color_bgr
        self.add_occlusion_barrier: bool = add_occlusion_barrier

        # Default occlusion wall in the center
        if barrier_rect is None:
            self.barrier_rect: Tuple[int, int, int, int] = (
                width // 2 - 45,
                height // 4,
                90,
                height // 2,
            )
        else:
            self.barrier_rect = barrier_rect

        # State initialization
        self.frame_idx: int = 0
        self.x: float = float(width // 4)
        self.y: float = float(height // 2)
        self.vx: float = 180.0  # px/s
        self.vy: float = 120.0  # px/s

        # Parametric angles for curves
        self.theta: float = 0.0

    def read(self) -> Tuple[bool, np.ndarray, Tuple[float, float], Tuple[float, float], bool]:
        """
        Generate the next video frame and associated ground truth state.

        Returns:
            (success, frame_bgr, (gt_x, gt_y), (gt_vx, gt_vy), is_occluded)
        """
        self.frame_idx += 1
        t = self.frame_idx * self.dt

        # Kinematics propagation based on trajectory profile
        if self.trajectory == "linear":
            self.x += self.vx * self.dt
            self.y += self.vy * self.dt

            # Wall bounces
            if self.x - self.ball_radius <= 20 or self.x + self.ball_radius >= self.width - 20:
                self.vx = -self.vx
                self.x = np.clip(self.x, self.ball_radius + 21, self.width - self.ball_radius - 21)
            if self.y - self.ball_radius <= 20 or self.y + self.ball_radius >= self.height - 20:
                self.vy = -self.vy
                self.y = np.clip(self.y, self.ball_radius + 21, self.height - self.ball_radius - 21)

            curr_vx, curr_vy = self.vx, self.vy

        elif self.trajectory == "circular":
            center_x, center_y = self.width / 2.0, self.height / 2.0
            radius_x, radius_y = 220.0, 160.0
            omega = 1.2  # rad/s

            self.theta = omega * t
            self.x = center_x + radius_x * np.cos(self.theta)
            self.y = center_y + radius_y * np.sin(self.theta)

            curr_vx = -radius_x * omega * np.sin(self.theta)
            curr_vy = radius_y * omega * np.cos(self.theta)

        elif self.trajectory == "sine":
            # Sinusoidal wave traveling horizontally
            center_y = self.height / 2.0
            amplitude = 140.0
            freq = 1.5

            self.x += self.vx * self.dt
            if self.x - self.ball_radius <= 30 or self.x + self.ball_radius >= self.width - 30:
                self.vx = -self.vx
            self.y = center_y + amplitude * np.sin(freq * t)

            curr_vx = self.vx
            curr_vy = amplitude * freq * np.cos(freq * t)

        else:  # 'mixed' trajectory: smooth compound Lissajous / harmonic motion
            center_x, center_y = self.width / 2.0, self.height / 2.0
            ax, ay = 250.0, 180.0
            wx, wy = 0.9, 1.35
            phase = np.pi / 4.0

            self.x = center_x + ax * np.sin(wx * t)
            self.y = center_y + ay * np.sin(wy * t + phase)

            curr_vx = ax * wx * np.cos(wx * t)
            curr_vy = ay * wy * np.cos(wy * t + phase)

        # Check occlusion by the rectangular wall
        is_occluded = False
        if self.add_occlusion_barrier:
            bx, by, bw, bh = self.barrier_rect
            if (bx <= self.x <= bx + bw) and (by <= self.y <= by + bh):
                is_occluded = True

        # Render frame canvas
        frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)

        # Render background grid
        grid_step = 50
        for gx in range(0, self.width, grid_step):
            cv2.line(frame, (gx, 0), (gx, self.height), (230, 230, 230), 1)
        for gy in range(0, self.height, grid_step):
            cv2.line(frame, (0, gy), (self.width, gy), (230, 230, 230), 1)

        # Render ball if not occluded (or behind barrier)
        ball_center = (int(round(self.x)), int(round(self.y)))

        # Always draw ball shadow
        shadow_center = (ball_center[0] + 6, ball_center[1] + 8)
        cv2.circle(frame, shadow_center, self.ball_radius, (200, 200, 200), -1, cv2.LINE_AA)

        # Ball body with radial gradient / highlight
        cv2.circle(frame, ball_center, self.ball_radius, self.ball_color, -1, cv2.LINE_AA)
        highlight_center = (ball_center[0] - 6, ball_center[1] - 6)
        cv2.circle(frame, highlight_center, max(3, self.ball_radius // 3), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, ball_center, self.ball_radius, (0, 90, 180), 2, cv2.LINE_AA)

        # Render occlusion barrier wall ON TOP of the ball
        if self.add_occlusion_barrier:
            bx, by, bw, bh = self.barrier_rect
            # Wall shadow
            cv2.rectangle(frame, (bx + 4, by + 4), (bx + bw + 4, by + bh + 4), (180, 180, 180), -1)
            # Wall body
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 30, 30), 2)
            # Wall stripes pattern
            for sx in range(bx, bx + bw, 20):
                cv2.line(frame, (sx, by), (sx + 10, by + bh), (90, 90, 90), 1)
            cv2.putText(
                frame,
                "OCCLUSION WALL",
                (bx + 8, by + bh // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        gt_pos = (self.x, self.y)
        gt_vel = (curr_vx, curr_vy)
        return True, frame, gt_pos, gt_vel, is_occluded


def create_sample_video(
    output_video_path: Union[str, Path] = "data/sample_video.mp4",
    output_gt_csv: Optional[Union[str, Path]] = "data/sample_ground_truth.csv",
    num_frames: int = 360,
    width: int = 800,
    height: int = 600,
    fps: float = 30.0,
    trajectory: str = "mixed",
    add_occlusion_barrier: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Generate and save a benchmark MP4 video along with its ground-truth state CSV.

    Returns:
        Tuple of (video_path, gt_csv_path)
    """
    video_path = Path(output_video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    stream = SyntheticStream(
        width=width,
        height=height,
        fps=fps,
        trajectory=trajectory,
        add_occlusion_barrier=add_occlusion_barrier,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    gt_records: List[Dict[str, Any]] = []

    for idx in range(num_frames):
        ret, frame, gt_pos, gt_vel, is_occluded = stream.read()
        if not ret:
            break
        out.write(frame)

        if output_gt_csv:
            gt_records.append({
                "frame_index": idx + 1,
                "timestamp_s": (idx + 1) / fps,
                "gt_x": gt_pos[0],
                "gt_y": gt_pos[1],
                "gt_vx": gt_vel[0],
                "gt_vy": gt_vel[1],
                "is_occluded": int(is_occluded),
            })

    out.release()

    gt_path: Optional[Path] = None
    if output_gt_csv and gt_records:
        gt_path = Path(output_gt_csv)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_index", "timestamp_s", "gt_x", "gt_y", "gt_vx", "gt_vy", "is_occluded"]
            )
            writer.writeheader()
            writer.writerows(gt_records)

    return video_path, gt_path


if __name__ == "__main__":
    v_path, csv_path = create_sample_video()
    print(f"Sample benchmark video created: {v_path}")
    if csv_path:
        print(f"Sample ground-truth CSV created: {csv_path}")
