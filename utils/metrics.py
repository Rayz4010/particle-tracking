"""
Tracking Evaluation Metrics and Telemetry Logger.

Calculates:
- Root Mean Square Error (RMSE) for position and velocity
- Mean Absolute Error (MAE) and Peak Error
- High-resolution execution latency profiling (mean, median, p95, max)
- Occlusion recovery dynamics (time to convergence post-occlusion)
- CSV logging and tabulated summary reporting
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import csv
import numpy as np


@dataclass
class FrameRecord:
    """Telemetry data captured for a single video frame."""
    frame_index: int
    timestamp_s: float
    is_detected: bool
    is_occluded: bool
    gt_pos: Optional[Tuple[float, float]] = None
    gt_vel: Optional[Tuple[float, float]] = None
    meas_pos: Optional[Tuple[float, float]] = None
    kf_pos: Tuple[float, float] = (0.0, 0.0)
    kf_vel: Tuple[float, float] = (0.0, 0.0)
    pf_pos: Tuple[float, float] = (0.0, 0.0)
    pf_vel: Tuple[float, float] = (0.0, 0.0)
    kf_error: float = 0.0
    pf_error: float = 0.0
    kf_latency_ms: float = 0.0
    pf_latency_ms: float = 0.0
    detector_latency_ms: float = 0.0
    pf_n_eff: float = 0.0


class TrackingMetrics:
    """
    Online and batch metrics logger for comparative tracking benchmark.
    """

    def __init__(self, recovery_threshold_px: float = 15.0) -> None:
        """
        Args:
            recovery_threshold_px: Error threshold (pixels) below which a tracker
                                   is considered successfully recovered after occlusion.
        """
        self.recovery_threshold_px: float = float(recovery_threshold_px)
        self.records: List[FrameRecord] = []

    def record_frame(
        self,
        frame_index: int,
        timestamp_s: float,
        is_detected: bool,
        is_occluded: bool,
        gt_pos: Optional[Tuple[float, float]],
        meas_pos: Optional[Tuple[float, float]],
        kf_pos: Tuple[float, float],
        kf_vel: Tuple[float, float],
        pf_pos: Tuple[float, float],
        pf_vel: Tuple[float, float],
        kf_latency_ms: float,
        pf_latency_ms: float,
        detector_latency_ms: float = 0.0,
        gt_vel: Optional[Tuple[float, float]] = None,
        pf_n_eff: float = 0.0,
    ) -> FrameRecord:
        """Record telemetry for a single frame and compute instant errors."""
        # Reference point for error: prefer ground truth if available, else valid measurement
        ref_pos = gt_pos if gt_pos is not None else meas_pos

        if ref_pos is not None:
            rx, ry = ref_pos
            kf_err = float(np.hypot(kf_pos[0] - rx, kf_pos[1] - ry))
            pf_err = float(np.hypot(pf_pos[0] - rx, pf_pos[1] - ry))
        else:
            kf_err = 0.0
            pf_err = 0.0

        rec = FrameRecord(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            is_detected=is_detected,
            is_occluded=is_occluded,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            meas_pos=meas_pos,
            kf_pos=kf_pos,
            kf_vel=kf_vel,
            pf_pos=pf_pos,
            pf_vel=pf_vel,
            kf_error=kf_err,
            pf_error=pf_err,
            kf_latency_ms=kf_latency_ms,
            pf_latency_ms=pf_latency_ms,
            detector_latency_ms=detector_latency_ms,
            pf_n_eff=pf_n_eff,
        )
        self.records.append(rec)
        return rec

    def compute_summary(self) -> Dict[str, Any]:
        """
        Calculate aggregate quantitative benchmarking statistics.
        """
        if not self.records:
            return {}

        total_frames = len(self.records)
        kf_latencies = np.array([r.kf_latency_ms for r in self.records], dtype=np.float64)
        pf_latencies = np.array([r.pf_latency_ms for r in self.records], dtype=np.float64)
        det_latencies = np.array([r.detector_latency_ms for r in self.records], dtype=np.float64)

        # Separate indices for all frames, visible frames, and occluded frames
        valid_eval_indices = [
            i for i, r in enumerate(self.records)
            if (r.gt_pos is not None or r.meas_pos is not None)
        ]
        visible_indices = [
            i for i, r in enumerate(self.records)
            if (r.gt_pos is not None or r.meas_pos is not None) and not r.is_occluded
        ]
        occluded_indices = [
            i for i, r in enumerate(self.records)
            if (r.gt_pos is not None or r.meas_pos is not None) and r.is_occluded
        ]

        def calc_error_stats(errors: np.ndarray) -> Dict[str, float]:
            if len(errors) == 0:
                return {"rmse": 0.0, "mae": 0.0, "max": 0.0, "std": 0.0}
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mae = float(np.mean(errors))
            max_err = float(np.max(errors))
            std_err = float(np.std(errors))
            return {"rmse": rmse, "mae": mae, "max": max_err, "std": std_err}

        all_kf_err = np.array([self.records[i].kf_error for i in valid_eval_indices])
        all_pf_err = np.array([self.records[i].pf_error for i in valid_eval_indices])

        vis_kf_err = np.array([self.records[i].kf_error for i in visible_indices])
        vis_pf_err = np.array([self.records[i].pf_error for i in visible_indices])

        occ_kf_err = np.array([self.records[i].kf_error for i in occluded_indices])
        occ_pf_err = np.array([self.records[i].pf_error for i in occluded_indices])

        kf_stats_all = calc_error_stats(all_kf_err)
        pf_stats_all = calc_error_stats(all_pf_err)
        kf_stats_vis = calc_error_stats(vis_kf_err)
        pf_stats_vis = calc_error_stats(vis_pf_err)
        kf_stats_occ = calc_error_stats(occ_kf_err)
        pf_stats_occ = calc_error_stats(occ_pf_err)

        # Occlusion Recovery Analysis
        kf_rec_times, pf_rec_times = self._analyze_recovery_times()

        def calc_latency_stats(latencies: np.ndarray) -> Dict[str, float]:
            return {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "std_ms": float(np.std(latencies)),
                "fps_equivalent": float(1000.0 / max(np.mean(latencies), 1e-4)),
            }

        summary = {
            "total_frames": total_frames,
            "visible_frames": len(visible_indices),
            "occluded_frames": len(occluded_indices),
            "kalman_filter": {
                "overall": kf_stats_all,
                "visible": kf_stats_vis,
                "occluded": kf_stats_occ,
                "latency": calc_latency_stats(kf_latencies),
                "mean_recovery_frames": float(np.mean(kf_rec_times)) if kf_rec_times else 0.0,
                "recovery_episodes": len(kf_rec_times),
            },
            "particle_filter": {
                "overall": pf_stats_all,
                "visible": pf_stats_vis,
                "occluded": pf_stats_occ,
                "latency": calc_latency_stats(pf_latencies),
                "mean_recovery_frames": float(np.mean(pf_rec_times)) if pf_rec_times else 0.0,
                "recovery_episodes": len(pf_rec_times),
            },
            "detector_latency": calc_latency_stats(det_latencies) if len(det_latencies) > 0 else {},
        }
        return summary

    def _analyze_recovery_times(self) -> Tuple[List[int], List[int]]:
        """
        Identify occlusion transitions (occluded -> unoccluded) and compute
        frames needed for each tracker's error to recover below threshold.
        """
        kf_recovery_frames: List[int] = []
        pf_recovery_frames: List[int] = []

        in_occlusion = False
        reappear_idx: Optional[int] = None

        for i, rec in enumerate(self.records):
            if rec.is_occluded and not in_occlusion:
                in_occlusion = True
            elif not rec.is_occluded and in_occlusion:
                in_occlusion = False
                reappear_idx = i

                # Inspect subsequent frames for KF recovery
                kf_recovered_in = -1
                for j in range(reappear_idx, len(self.records)):
                    if self.records[j].is_occluded:
                        break  # Hit another occlusion before recovering
                    if self.records[j].kf_error <= self.recovery_threshold_px:
                        kf_recovered_in = j - reappear_idx
                        break
                if kf_recovered_in != -1:
                    kf_recovery_frames.append(kf_recovered_in)

                # Inspect subsequent frames for PF recovery
                pf_recovered_in = -1
                for j in range(reappear_idx, len(self.records)):
                    if self.records[j].is_occluded:
                        break
                    if self.records[j].pf_error <= self.recovery_threshold_px:
                        pf_recovered_in = j - reappear_idx
                        break
                if pf_recovered_in != -1:
                    pf_recovery_frames.append(pf_recovered_in)

        return kf_recovery_frames, pf_recovery_frames

    def export_csv(self, filepath: Union[str, Path]) -> None:
        """Export frame-level telemetry to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame_index", "timestamp_s", "is_detected", "is_occluded",
            "gt_x", "gt_y", "meas_x", "meas_y",
            "kf_x", "kf_y", "kf_vx", "kf_vy", "kf_error_px", "kf_latency_ms",
            "pf_x", "pf_y", "pf_vx", "pf_vy", "pf_error_px", "pf_latency_ms",
            "detector_latency_ms", "pf_n_eff"
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow({
                    "frame_index": r.frame_index,
                    "timestamp_s": f"{r.timestamp_s:.4f}",
                    "is_detected": int(r.is_detected),
                    "is_occluded": int(r.is_occluded),
                    "gt_x": f"{r.gt_pos[0]:.2f}" if r.gt_pos else "",
                    "gt_y": f"{r.gt_pos[1]:.2f}" if r.gt_pos else "",
                    "meas_x": f"{r.meas_pos[0]:.2f}" if r.meas_pos else "",
                    "meas_y": f"{r.meas_pos[1]:.2f}" if r.meas_pos else "",
                    "kf_x": f"{r.kf_pos[0]:.2f}",
                    "kf_y": f"{r.kf_pos[1]:.2f}",
                    "kf_vx": f"{r.kf_vel[0]:.2f}",
                    "kf_vy": f"{r.kf_vel[1]:.2f}",
                    "kf_error_px": f"{r.kf_error:.2f}",
                    "kf_latency_ms": f"{r.kf_latency_ms:.4f}",
                    "pf_x": f"{r.pf_pos[0]:.2f}",
                    "pf_y": f"{r.pf_pos[1]:.2f}",
                    "pf_vx": f"{r.pf_vel[0]:.2f}",
                    "pf_vy": f"{r.pf_vel[1]:.2f}",
                    "pf_error_px": f"{r.pf_error:.2f}",
                    "pf_latency_ms": f"{r.pf_latency_ms:.4f}",
                    "detector_latency_ms": f"{r.detector_latency_ms:.4f}",
                    "pf_n_eff": f"{r.pf_n_eff:.1f}",
                })

    def format_table(self) -> str:
        """Return formatted console summary table."""
        s = self.compute_summary()
        if not s:
            return "No tracking metrics recorded."

        kf = s["kalman_filter"]
        pf = s["particle_filter"]

        lines = [
            "=" * 76,
            f"{'BENCHMARK SUMMARY: KALMAN FILTER vs. PARTICLE FILTER':^76}",
            "=" * 76,
            f"Total Frames: {s['total_frames']} | Visible: {s['visible_frames']} | Occluded: {s['occluded_frames']}",
            "-" * 76,
            f"{'Metric':<30} | {'Kalman Filter':<20} | {'Particle Filter':<20}",
            "-" * 76,
            f"{'Overall RMSE (px)':<30} | {kf['overall']['rmse']:<20.2f} | {pf['overall']['rmse']:<20.2f}",
            f"{'Visible-only RMSE (px)':<30} | {kf['visible']['rmse']:<20.2f} | {pf['visible']['rmse']:<20.2f}",
            f"{'Occlusion RMSE (px)':<30} | {kf['occluded']['rmse']:<20.2f} | {pf['occluded']['rmse']:<20.2f}",
            f"{'Overall MAE (px)':<30} | {kf['overall']['mae']:<20.2f} | {pf['overall']['mae']:<20.2f}",
            f"{'Max Error (px)':<30} | {kf['overall']['max']:<20.2f} | {pf['overall']['max']:<20.2f}",
            "-" * 76,
            f"{'Mean Latency (ms)':<30} | {kf['latency']['mean_ms']:<20.3f} | {pf['latency']['mean_ms']:<20.3f}",
            f"{'Median Latency (ms)':<30} | {kf['latency']['median_ms']:<20.3f} | {pf['latency']['median_ms']:<20.3f}",
            f"{'P95 Latency (ms)':<30} | {kf['latency']['p95_ms']:<20.3f} | {pf['latency']['p95_ms']:<20.3f}",
            f"{'Theoretical Filter FPS':<30} | {kf['latency']['fps_equivalent']:<20.1f} | {pf['latency']['fps_equivalent']:<20.1f}",
            "-" * 76,
            f"{'Mean Occlusion Recovery (frames)':<30} | {kf['mean_recovery_frames']:<20.2f} | {pf['mean_recovery_frames']:<20.2f}",
            "=" * 76,
        ]
        return "\n".join(lines)
