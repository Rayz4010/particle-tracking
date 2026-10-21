"""
Real-Time Object Tracking Application: Kalman Filter vs. Particle Filter Comparison.

Interactive CLI application:
- Dual/split-screen and overlay visualization
- Real-time telemetry HUD (FPS, algorithm latency, error metrics, occlusion status)
- Color blob detection with interactive ROI calibration
- Simulated occlusions (manual toggle, periodic dropouts, and spatial barriers)
- Interactive key controls:
    [q]: Quit
    [s]: Select ROI / calibrate color
    [o]: Toggle simulated occlusion
    [p]: Pause / Resume playback
    [c]: Clear trajectory trail history
    [m]: Toggle display mode (Overlay <-> Split-Screen)
    [r]: Reset tracker filters
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Optional, Tuple
import cv2
import numpy as np

from data.synthetic_generator import SyntheticStream
from detectors.detector import ColorBlobDetector, DetectionResult
from trackers.kalman_filter import KalmanFilterTracker
from trackers.particle_filter import ParticleFilterTracker
from utils.metrics import TrackingMetrics
from utils.visualizer import RenderMode, TrackingVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Object Tracking: Kalman Filter vs. Particle Filter Comparison"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/sample_video.mp4",
        help="Input video source: webcam index (e.g. '0'), video file path, or 'synthetic'.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="orange",
        choices=["orange", "green", "blue", "red", "yellow", "cyan"],
        help="Target color preset for blob detector.",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Prompt interactive ROI selection on the initial video frame.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=1000,
        help="Number of particles for Particle Filter (default: 1000).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Synthetic Gaussian measurement noise standard deviation in pixels (default: 0.0).",
    )
    parser.add_argument(
        "--occlusion-freq",
        type=int,
        default=0,
        help="Periodic occlusion cycle interval in frames (0 = disabled).",
    )
    parser.add_argument(
        "--occlusion-duration",
        type=int,
        default=30,
        help="Duration of periodic occlusion in frames (default: 30).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overlay",
        choices=["overlay", "split"],
        help="Visualization mode: 'overlay' or 'split' screen (default: overlay).",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to save annotated output video (MP4 format).",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Path to export telemetry metrics to CSV.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without opening GUI windows (for servers/benchmarks).",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive Streamlit web dashboard in the browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.web:
        import subprocess
        app_file = Path(__file__).resolve().parent / "app.py"
        print(f"Launching Streamlit web dashboard: {app_file}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])
        return

    # Determine input stream type
    is_synthetic = args.source.lower() == "synthetic"
    is_webcam = False
    cap = None
    synth_stream = None
    fps = 30.0

    if is_synthetic:
        synth_stream = SyntheticStream(
            width=800,
            height=600,
            fps=fps,
            trajectory="mixed",
            add_occlusion_barrier=True,
        )
        print("Using Synthetic Stream with moving ball and central occlusion barrier.")
    else:
        # Check if source is an integer webcam index
        try:
            cam_idx = int(args.source)
            is_webcam = True
            print(f"Connecting to Webcam index {cam_idx}...")
            # Prefer AVFoundation on macOS if available, else default backend
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                print(f"\n[Error] Unable to open webcam at index {cam_idx}.")
                print("On macOS, please verify camera permission is enabled:")
                print("  System Settings -> Privacy & Security -> Camera")
                print("Ensure Terminal / your IDE is granted access, then retry.\n")
                sys.exit(1)

            # Camera warm-up frames to stabilize auto-exposure and white balance
            print(f"Webcam index {cam_idx} opened. Stabilizing camera sensor...")
            for _ in range(6):
                cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            print(f"Webcam sensor ready ({fps:.1f} FPS).")
        except ValueError:
            src_path = Path(args.source)
            if not src_path.exists():
                print(f"[Warning] Source '{args.source}' not found. Falling back to synthetic stream.")
                is_synthetic = True
                synth_stream = SyntheticStream(
                    width=800,
                    height=600,
                    fps=fps,
                    trajectory="mixed",
                    add_occlusion_barrier=True,
                )
            else:
                cap = cv2.VideoCapture(str(src_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                print(f"Opened video file: {src_path} ({fps:.1f} FPS)")

    # Read first frame to ascertain dimensions and initial setup
    if is_synthetic:
        assert synth_stream is not None
        ret, first_frame, gt_pos, gt_vel, s_occ = synth_stream.read()
    else:
        assert cap is not None
        ret, first_frame = cap.read()

    if not ret or first_frame is None:
        print("[Error] Failed to read initial frame from video source. Exiting.")
        sys.exit(1)

    frame_h, frame_w = first_frame.shape[:2]
    dt = 1.0 / fps

    # Detector setup
    detector = ColorBlobDetector(
        color_preset=args.color,
        measurement_noise_std=args.noise,
        periodic_occlusion_interval=args.occlusion_freq,
        periodic_occlusion_duration=args.occlusion_duration,
    )

    # Initialize Visualizer
    render_mode = RenderMode.SPLIT_SCREEN if args.mode == "split" else RenderMode.OVERLAY
    visualizer = TrackingVisualizer(trail_length=70, render_mode=render_mode)

    # Interactive non-blocking mouse ROI selection state
    is_in_roi_mode: bool = bool(args.select_roi and not args.headless)
    mouse_dragging: bool = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_current: Optional[Tuple[int, int]] = None
    pending_roi: Optional[Tuple[int, int, int, int]] = None
    toast_message: str = ""
    toast_timer: int = 0

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal mouse_dragging, drag_start, drag_current, pending_roi, is_in_roi_mode
        # In split-screen mode, normalize x coordinate to frame_w
        norm_x = x if visualizer.render_mode == RenderMode.OVERLAY else (x % frame_w)
        norm_y = y

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_dragging = True
            drag_start = (norm_x, norm_y)
            drag_current = (norm_x, norm_y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse_dragging:
            drag_current = (norm_x, norm_y)
        elif event == cv2.EVENT_LBUTTONUP and mouse_dragging:
            mouse_dragging = False
            if drag_start is not None:
                x0, y0 = drag_start
                rx = min(x0, norm_x)
                ry = min(y0, norm_y)
                rw = abs(norm_x - x0)
                rh = abs(norm_y - y0)
                if rw > 12 and rh > 12:
                    pending_roi = (rx, ry, rw, rh)
            drag_start = None
            drag_current = None

    window_name = "Object Tracking: Kalman Filter vs. Particle Filter"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 750)
        cv2.setMouseCallback(window_name, on_mouse)

    if is_in_roi_mode:
        print("\n[Target Selection Mode ACTIVE] Click and drag on any object in the live window to lock onto it.")

    # Initialize Trackers
    kf = KalmanFilterTracker(
        dt=dt,
        process_noise_std=8.0,
        measurement_noise_std=max(3.0, args.noise),
    )
    pf = ParticleFilterTracker(
        num_particles=args.particles,
        dt=dt,
        process_noise_pos=4.0,
        process_noise_vel=25.0,
        measurement_noise_std=max(12.0, args.noise * 2.0),
        resample_threshold_ratio=0.5,
        resample_method="systematic",
    )

    metrics = TrackingMetrics(recovery_threshold_px=15.0)

    # Video recording writer setup if requested
    video_writer = None
    if args.save_video:
        save_path = Path(args.save_video)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Determine canvas size based on render mode
        out_w = frame_w * 2 if visualizer.render_mode == RenderMode.SPLIT_SCREEN else frame_w
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (out_w, frame_h))
        print(f"[Recording] Output video will be saved to: {save_path}")

    # Telemetry and loop variables
    frame_index = 0
    trackers_initialized = False
    is_paused = False
    fps_smooth = fps
    t_prev = time.perf_counter()

    print("\n--------------------------------------------------------------")
    print(" TRACKER RUNNING. Press 'q' to quit, 's' to select ROI.")
    print(" Controls: [q] Quit  [s] ROI  [o] Occlusion  [p] Pause  [m] Mode  [c] Clear")
    print("--------------------------------------------------------------\n")

    current_frame = first_frame
    current_gt_pos: Optional[Tuple[float, float]] = None
    current_gt_vel: Optional[Tuple[float, float]] = None
    sim_occ = False

    while True:
        if not is_paused:
            frame_index += 1
            if frame_index > 1:
                if is_synthetic:
                    assert synth_stream is not None
                    ret, current_frame, current_gt_pos, current_gt_vel, sim_occ = synth_stream.read()
                    if not ret:
                        print("Synthetic stream sequence finished.")
                        break
                else:
                    assert cap is not None
                    ret, current_frame = cap.read()
                    if not ret or current_frame is None:
                        if is_webcam:
                            print("Camera disconnected.")
                        else:
                            print("Video playback reached end.")
                        break

            # Process interactive ROI selection if user dragged mouse
            if pending_roi is not None:
                rx, ry, rw, rh = pending_roi
                pending_roi = None
                rx = max(0, min(frame_w - 5, rx))
                ry = max(0, min(frame_h - 5, ry))
                rw = max(5, min(frame_w - rx, rw))
                rh = max(5, min(frame_h - ry, rh))
                detector.calibrate_from_roi(current_frame, (rx, ry, rw, rh))
                kf.reset()
                pf.reset()
                trackers_initialized = False
                visualizer.clear_trails()
                is_in_roi_mode = False
                toast_message = f"TARGET LOCKED! Calibrated ({rw}x{rh} px)"
                toast_timer = 70
                print(f"\n[Target Locked] Auto-calibrated from ROI ({rx}, {ry}, {rw}, {rh}).")

            # 1. Measurement Detection
            t_det0 = time.perf_counter()
            det_res = detector.detect(current_frame)
            t_det_ms = (time.perf_counter() - t_det0) * 1000.0

            # Merge synthetic barrier occlusion if using synthetic stream
            if sim_occ:
                det_res.is_occluded = True
                det_res.centroid = None

            is_target_lost = det_res.is_occluded or (not det_res.is_detected)
            meas = None if is_target_lost else det_res.centroid

            # Initialization upon first clear detection
            if not trackers_initialized:
                if meas is not None:
                    kf.initialize(meas)
                    pf.initialize(meas, init_pos_std=10.0, init_vel_std=60.0)
                    trackers_initialized = True
                    print(f"Target locked at frame {frame_index}: {meas}")

            # 2. Kalman Filter Predict & Update
            t_kf0 = time.perf_counter()
            if trackers_initialized:
                kf.predict(dt=dt)
                kf.update(meas)
            t_kf_ms = (time.perf_counter() - t_kf0) * 1000.0

            # 3. Particle Filter Predict & Update
            t_pf0 = time.perf_counter()
            if trackers_initialized:
                pf.predict(dt=dt)
                pf.update(meas)
            t_pf_ms = (time.perf_counter() - t_pf0) * 1000.0

            # Real-time FPS calculation
            t_now = time.perf_counter()
            instant_fps = 1.0 / max(1e-4, (t_now - t_prev))
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            t_prev = t_now

            # Telemetry logging
            record = None
            if trackers_initialized:
                record = metrics.record_frame(
                    frame_index=frame_index,
                    timestamp_s=frame_index * dt,
                    is_detected=det_res.is_detected,
                    is_occluded=det_res.is_occluded,
                    gt_pos=current_gt_pos,
                    gt_vel=current_gt_vel,
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

        # 4. Visualization Rendering
        if trackers_initialized:
            kf_state = kf.get_state()
            kf_cov = kf.get_covariance()
            pf_state = pf.get_state()
            pf_parts = pf.get_particles(sample_size=300)
        else:
            kf_state = np.zeros(4)
            kf_cov = None
            pf_state = np.zeros(4)
            pf_parts = (np.empty((0, 2)), np.empty(0))

        vis_frame = visualizer.render_frame(
            frame=current_frame,
            detection=det_res,
            kf_state=kf_state,
            kf_cov=kf_cov,
            pf_state=pf_state,
            pf_particles=pf_parts,
            fps=fps_smooth,
            record=record,
            is_manual_occlusion=detector.is_manual_occlusion,
            spatial_box=detector.spatial_occlusion_box,
        )

        # Draw live mouse drag selection box
        if mouse_dragging and drag_start and drag_current:
            x0, y0 = drag_start
            x1, y1 = drag_current
            bx = min(x0, x1)
            by = min(y0, y1)
            bw = abs(x1 - x0)
            bh = abs(y1 - y0)
            cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.drawMarker(vis_frame, (bx + bw // 2, by + bh // 2), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(vis_frame, f"SELECT: {bw}x{bh} (Release to lock)", (bx, max(22, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw ROI selection mode banner
        if is_in_roi_mode:
            banner_overlay = vis_frame.copy()
            cv2.rectangle(banner_overlay, (0, vis_frame.shape[0] - 62), (vis_frame.shape[1], vis_frame.shape[0] - 28), (0, 100, 220), -1)
            cv2.addWeighted(banner_overlay, 0.75, vis_frame, 0.25, 0, vis_frame)
            cv2.putText(
                vis_frame,
                "TARGET SELECTION: Click and drag with mouse over any object to lock  |  Press 's' to cancel",
                (14, vis_frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw toast confirmation banner if active
        if toast_timer > 0:
            toast_timer -= 1
            toast_overlay = vis_frame.copy()
            cv2.rectangle(toast_overlay, (0, 88), (vis_frame.shape[1], 118), (30, 140, 30), -1)
            cv2.addWeighted(toast_overlay, 0.75, vis_frame, 0.25, 0, vis_frame)
            cv2.putText(vis_frame, toast_message, (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if video_writer is not None:
            video_writer.write(vis_frame)

        if not args.headless:
            cv2.imshow(window_name, vis_frame)
            key = cv2.waitKey(1 if not is_paused else 30) & 0xFF

            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\nQuit requested by user.")
                break
            elif key == ord("p"):  # Pause/Resume
                is_paused = not is_paused
                print(f"Playback {'Paused' if is_paused else 'Resumed'}.")
            elif key == ord("o"):  # Toggle simulated occlusion
                is_occ = detector.toggle_manual_occlusion()
                print(f"Manual Occlusion: {'ON' if is_occ else 'OFF'}")
            elif key == ord("c"):  # Clear trail history
                visualizer.clear_trails()
                print("Trajectory history cleared.")
            elif key == ord("m"):  # Toggle display mode
                new_mode = visualizer.toggle_mode()
                print(f"Switched display mode to: {new_mode.value.upper()}")
            elif key == ord("r"):  # Reset filters
                kf.reset()
                pf.reset()
                trackers_initialized = False
                visualizer.clear_trails()
                print("Filters reset. Awaiting new target detection...")
            elif key == ord("s"):  # Toggle ROI selection mode
                is_in_roi_mode = not is_in_roi_mode
                if is_in_roi_mode:
                    print("\n[Target Selection Mode ACTIVE] Click and drag mouse over any object on the live video.")
                else:
                    print("\n[Target Selection Mode CANCELLED]")

    # Cleanup resources
    if cap is not None:
        cap.release()
    if video_writer is not None:
        video_writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # Print final telemetry summary
    print("\n" + metrics.format_table())

    if args.save_csv:
        metrics.export_csv(args.save_csv)
        print(f"Metrics CSV exported to: {args.save_csv}")


if __name__ == "__main__":
    main()
