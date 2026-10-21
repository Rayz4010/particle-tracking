"""
Evaluation & Benchmarking Pipeline for Kalman Filter vs. Particle Filter.

Runs headless comparative evaluation across real or synthetic video sequences.
Computes:
- RMSE & MAE against Ground Truth / Detector
- Per-frame computation latency statistics (mean, median, p95, max)
- Occlusion recovery dynamics
- Exports telemetry to CSV and publishes publication-grade Matplotlib visual charts:
  * rmse_comparison.png
  * latency_boxplot.png
  * occlusion_recovery.png
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np

from data.synthetic_generator import SyntheticStream, create_sample_video
from detectors.detector import ColorBlobDetector, DetectionResult
from trackers.kalman_filter import KalmanFilterTracker
from trackers.particle_filter import ParticleFilterTracker
from utils.metrics import FrameRecord, TrackingMetrics


def run_benchmark(
    video_source: Optional[str] = None,
    gt_csv_path: Optional[str] = None,
    num_frames: int = 360,
    num_particles: int = 1000,
    synthetic_noise_std: float = 3.0,
    trajectory_type: str = "mixed",
    output_dir: str = "benchmark_output",
) -> TrackingMetrics:
    """
    Execute offline headless benchmark comparing Kalman Filter and Particle Filter.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================================")
    print(f" STARTING BENCHMARK: Kalman Filter vs. Particle Filter (N={num_particles})")
    print(f" Measurement Noise: {synthetic_noise_std} px | Scenario: {trajectory_type}")
    print(f"==================================================================\n")

    # Determine video source
    using_synthetic_stream = False
    cap = None
    stream = None
    fps = 30.0

    if video_source and video_source.lower() != "synthetic" and Path(video_source).exists():
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Loaded video file: {video_source} ({fps:.1f} FPS)")
    else:
        using_synthetic_stream = True
        stream = SyntheticStream(
            width=800,
            height=600,
            fps=fps,
            trajectory=trajectory_type,
            add_occlusion_barrier=True,
        )
        print("Using synthetic continuous stream with central occlusion barrier.")

    # Load Ground Truth if available
    gt_lookup: Dict[int, Tuple[float, float, float, float]] = {}
    if gt_csv_path and Path(gt_csv_path).exists():
        import csv
        with open(gt_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                f_idx = int(row["frame_index"])
                gt_lookup[f_idx] = (
                    float(row["gt_x"]),
                    float(row["gt_y"]),
                    float(row["gt_vx"]),
                    float(row["gt_vy"]),
                    int(row.get("is_occluded", 0)),
                )
        print(f"Loaded ground-truth records for {len(gt_lookup)} frames.")

    # Initialize Detector
    detector = ColorBlobDetector(
        color_preset="orange",
        measurement_noise_std=synthetic_noise_std,
    )

    # Initialize Trackers
    dt = 1.0 / fps
    kf = KalmanFilterTracker(
        dt=dt,
        process_noise_std=8.0,
        measurement_noise_std=max(3.0, synthetic_noise_std),
    )
    pf = ParticleFilterTracker(
        num_particles=num_particles,
        dt=dt,
        process_noise_pos=4.0,
        process_noise_vel=25.0,
        measurement_noise_std=max(12.0, synthetic_noise_std * 2.0),
        resample_threshold_ratio=0.5,
        resample_method="systematic",
    )

    metrics = TrackingMetrics(recovery_threshold_px=15.0)

    frame_idx = 0
    trackers_initialized = False

    while frame_idx < num_frames:
        frame_idx += 1
        gt_pos: Optional[Tuple[float, float]] = None
        gt_vel: Optional[Tuple[float, float]] = None
        sim_is_occluded = False

        if using_synthetic_stream:
            assert stream is not None
            ret, frame, s_gt_pos, s_gt_vel, s_occ = stream.read()
            if not ret:
                break
            gt_pos = s_gt_pos
            gt_vel = s_gt_vel
            sim_is_occluded = s_occ
        else:
            assert cap is not None
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx in gt_lookup:
                gx, gy, gvx, gvy, gocc = gt_lookup[frame_idx]
                gt_pos = (gx, gy)
                gt_vel = (gvx, gvy)
                sim_is_occluded = bool(gocc)

        # 1. Detector Step
        t_det0 = time.perf_counter()
        det_result = detector.detect(frame)
        t_det_ms = (time.perf_counter() - t_det0) * 1000.0

        # Sync occlusion state
        is_occluded = det_result.is_occluded or sim_is_occluded or (not det_result.is_detected)
        meas = None if is_occluded else det_result.centroid

        # If not initialized yet, wait for first unoccluded detection
        if not trackers_initialized:
            if not is_occluded and det_result.centroid is not None:
                init_pt = det_result.centroid
                kf.initialize(init_pt)
                pf.initialize(init_pt, init_pos_std=10.0, init_vel_std=60.0)
                trackers_initialized = True
            continue

        # 2. Kalman Filter Predict & Update
        t_kf0 = time.perf_counter()
        kf.predict(dt=dt)
        kf.update(meas)
        t_kf_ms = (time.perf_counter() - t_kf0) * 1000.0

        # 3. Particle Filter Predict & Update
        t_pf0 = time.perf_counter()
        pf.predict(dt=dt)
        pf.update(meas)
        t_pf_ms = (time.perf_counter() - t_pf0) * 1000.0

        # Record Frame Telemetry
        metrics.record_frame(
            frame_index=frame_idx,
            timestamp_s=frame_idx * dt,
            is_detected=det_result.is_detected,
            is_occluded=is_occluded,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            meas_pos=meas,
            kf_pos=kf.get_position(),
            kf_vel=kf.get_velocity(),
            pf_pos=pf.get_position(),
            pf_vel=pf.get_velocity(),
            kf_latency_ms=t_kf_ms,
            pf_latency_ms=t_pf_ms,
            detector_latency_ms=t_det_ms,
            pf_n_eff=pf.n_eff,
        )

    if cap is not None:
        cap.release()

    # Print summary table
    print(metrics.format_table())

    # Export Telemetry CSV
    csv_file = out_path / "benchmark_results.csv"
    metrics.export_csv(csv_file)
    print(f"\n[Artifact] Telemetry CSV exported: {csv_file}")

    # Generate Comparison Figures
    generate_comparison_plots(metrics, out_path)

    return metrics


def generate_comparison_plots(metrics: TrackingMetrics, output_dir: Path) -> None:
    """
    Generate publication-ready Matplotlib comparison graphs:
    1. rmse_comparison.png: 2D Trajectory comparison & Estimation Error over time
    2. latency_boxplot.png: Execution latency distributions and boxplot
    3. occlusion_recovery.png: Transient response during and following occlusion
    """
    records = metrics.records
    if not records:
        return

    frames = [r.frame_index for r in records]
    kf_errs = [r.kf_error for r in records]
    pf_errs = [r.pf_error for r in records]
    kf_latencies = [r.kf_latency_ms for r in records]
    pf_latencies = [r.pf_latency_ms for r in records]
    is_occ = [r.is_occluded for r in records]

    has_gt = any(r.gt_pos is not None for r in records)
    gt_xs = [r.gt_pos[0] if r.gt_pos else np.nan for r in records]
    gt_ys = [r.gt_pos[1] if r.gt_pos else np.nan for r in records]
    kf_xs = [r.kf_pos[0] for r in records]
    kf_ys = [r.kf_pos[1] for r in records]
    pf_xs = [r.pf_pos[0] for r in records]
    pf_ys = [r.pf_pos[1] for r in records]

    # Set publication styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # =========================================================================
    # 1. RMSE & Trajectory Comparison (rmse_comparison.png)
    # =========================================================================
    fig, (ax_traj, ax_err) = plt.subplots(2, 1, figsize=(11, 9))

    # Top Plot: 2D Spatial Trajectory
    if has_gt:
        ax_traj.plot(gt_xs, gt_ys, "k--", label="Ground Truth", linewidth=2.0, alpha=0.85)
    ax_traj.plot(kf_xs, kf_ys, color="#0072B2", label="Kalman Filter", linewidth=1.8, alpha=0.9)
    ax_traj.plot(pf_xs, pf_ys, color="#D55E00", label="Particle Filter", linewidth=1.8, alpha=0.85)
    ax_traj.set_title("2D Trajectory Estimation Comparison", fontsize=12, fontweight="bold")
    ax_traj.set_xlabel("X Position (pixels)")
    ax_traj.set_ylabel("Y Position (pixels)")
    ax_traj.invert_yaxis()  # OpenCV image coordinates
    ax_traj.legend(loc="upper right", frameon=True)
    ax_traj.grid(True, linestyle=":", alpha=0.6)

    # Bottom Plot: Tracking Error over Time with Occlusion Shading
    ax_err.plot(frames, kf_errs, color="#0072B2", label="Kalman Filter Error", linewidth=1.5)
    ax_err.plot(frames, pf_errs, color="#D55E00", label="Particle Filter Error", linewidth=1.5)

    # Highlight occlusion regions with gray spans
    in_span = False
    start_idx = 0
    for i, occ in enumerate(is_occ):
        if occ and not in_span:
            in_span = True
            start_idx = frames[i]
        elif not occ and in_span:
            in_span = False
            ax_err.axvspan(start_idx, frames[i], color="gray", alpha=0.25, label="Occlusion Interval" if start_idx == frames[0] else "")
    if in_span:
        ax_err.axvspan(start_idx, frames[-1], color="gray", alpha=0.25)

    summary = metrics.compute_summary()
    kf_rmse = summary["kalman_filter"]["overall"]["rmse"]
    pf_rmse = summary["particle_filter"]["overall"]["rmse"]

    ax_err.set_title(
        f"Position Tracking Error Over Time (KF RMSE: {kf_rmse:.2f}px | PF RMSE: {pf_rmse:.2f}px)",
        fontsize=12, fontweight="bold"
    )
    ax_err.set_xlabel("Frame Index")
    ax_err.set_ylabel("Euclidean Error (pixels)")
    ax_err.grid(True, linestyle=":", alpha=0.6)

    # De-duplicate legend entries
    handles, labels = ax_err.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_err.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True)

    rmse_plot_path = output_dir / "rmse_comparison.png"
    fig.savefig(rmse_plot_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact] Chart generated: {rmse_plot_path}")

    # =========================================================================
    # 2. Execution Latency Boxplot & Histogram (latency_boxplot.png)
    # =========================================================================
    fig2, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(11, 5))

    # Boxplot
    box_data = [kf_latencies, pf_latencies]
    bp = ax_box.boxplot(
        box_data,
        tick_labels=["Kalman Filter", "Particle Filter"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    colors = ["#0072B2", "#D55E00"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    for mean in bp["means"]:
        mean.set(color="darkred", linewidth=2, linestyle="--")

    kf_lat = summary["kalman_filter"]["latency"]
    pf_lat = summary["particle_filter"]["latency"]

    ax_box.set_title("Filter Execution Latency Distribution", fontsize=11, fontweight="bold")
    ax_box.set_ylabel("Latency per Frame (ms)")
    ax_box.grid(True, linestyle=":", alpha=0.6)

    # Histogram / Density
    ax_hist.hist(kf_latencies, bins=25, alpha=0.6, color="#0072B2", label=f"KF (Mean: {kf_lat['mean_ms']:.3f}ms)")
    ax_hist.hist(pf_latencies, bins=25, alpha=0.6, color="#D55E00", label=f"PF (Mean: {pf_lat['mean_ms']:.3f}ms)")
    ax_hist.set_title("Latency Density Comparison", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Latency (ms)")
    ax_hist.set_ylabel("Frame Count")
    ax_hist.legend(loc="upper right", frameon=True)
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    latency_plot_path = output_dir / "latency_boxplot.png"
    fig2.savefig(latency_plot_path, dpi=200)
    plt.close(fig2)
    print(f"[Artifact] Chart generated: {latency_plot_path}")

    # =========================================================================
    # 3. Occlusion Recovery Dynamics (occlusion_recovery.png)
    # =========================================================================
    # Find longest occlusion episode for deep-dive analysis
    occ_episodes: List[Tuple[int, int]] = []
    in_ep = False
    ep_start = 0
    for i, occ in enumerate(is_occ):
        if occ and not in_ep:
            in_ep = True
            ep_start = i
        elif not occ and in_ep:
            in_ep = False
            occ_episodes.append((ep_start, i))
    if in_ep:
        occ_episodes.append((ep_start, len(is_occ) - 1))

    if occ_episodes:
        # Pick the episode with the most frames
        best_ep = max(occ_episodes, key=lambda ep: ep[1] - ep[0])
        # Include window before and after
        pad_before = max(0, best_ep[0] - 20)
        pad_after = min(len(records), best_ep[1] + 35)

        sub_frames = frames[pad_before:pad_after]
        sub_kf_err = kf_errs[pad_before:pad_after]
        sub_pf_err = pf_errs[pad_before:pad_after]

        fig3, ax_occ = plt.subplots(figsize=(10, 5))
        ax_occ.plot(sub_frames, sub_kf_err, color="#0072B2", marker="o", markersize=3, label="Kalman Filter Error", linewidth=1.8)
        ax_occ.plot(sub_frames, sub_pf_err, color="#D55E00", marker="s", markersize=3, label="Particle Filter Error", linewidth=1.8)

        # Highlight occlusion period
        ax_occ.axvspan(frames[best_ep[0]], frames[best_ep[1]], color="lightcoral", alpha=0.35, label="Occlusion Period (Target Hidden)")

        # Recovery threshold dashed line
        ax_occ.axhline(15.0, color="green", linestyle="--", linewidth=1.5, label="Recovery Threshold (15 px)")

        ax_occ.set_title("Transient Occlusion and Re-acquisition Dynamics", fontsize=12, fontweight="bold")
        ax_occ.set_xlabel("Frame Index")
        ax_occ.set_ylabel("Estimation Error (pixels)")
        ax_occ.legend(loc="upper right", frameon=True)
        ax_occ.grid(True, linestyle=":", alpha=0.6)

        recovery_plot_path = output_dir / "occlusion_recovery.png"
        fig3.savefig(recovery_plot_path, dpi=200)
        plt.close(fig3)
        print(f"[Artifact] Chart generated: {recovery_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmarking suite for Kalman Filter and Particle Filter object tracking."
    )
    parser.add_argument("--video", type=str, default=None, help="Input video path (defaults to synthetic)")
    parser.add_argument("--gt", type=str, default=None, help="Ground truth CSV file path")
    parser.add_argument("--frames", type=int, default=300, help="Number of benchmark frames to evaluate")
    parser.add_argument("--particles", type=int, default=1000, help="Number of particles for Particle Filter")
    parser.add_argument("--noise", type=float, default=4.0, help="Synthetic measurement noise std (pixels)")
    parser.add_argument("--trajectory", type=str, default="mixed", choices=["linear", "circular", "sine", "mixed"], help="Synthetic trajectory type")
    parser.add_argument("--output-dir", type=str, default="benchmark_output", help="Directory to save artifacts")

    args = parser.parse_args()

    # If default sample video exists and no video is explicitly passed, default to sample_video.mp4
    video_to_use = args.video
    gt_to_use = args.gt
    if video_to_use is None:
        if Path("data/sample_video.mp4").exists():
            video_to_use = "data/sample_video.mp4"
            if gt_to_use is None and Path("data/sample_ground_truth.csv").exists():
                gt_to_use = "data/sample_ground_truth.csv"
        else:
            video_to_use = "synthetic"
    elif video_to_use.lower() == "synthetic":
        gt_to_use = None

    run_benchmark(
        video_source=video_to_use,
        gt_csv_path=gt_to_use,
        num_frames=args.frames,
        num_particles=args.particles,
        synthetic_noise_std=args.noise,
        trajectory_type=args.trajectory,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
