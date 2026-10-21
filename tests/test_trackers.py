"""
Unit and Integration Tests for Real-Time Object Tracking Comparison Suite.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from detectors.detector import ColorBlobDetector, DetectionResult
from trackers.kalman_filter import KalmanFilterTracker
from trackers.particle_filter import ParticleFilterTracker
from utils.metrics import TrackingMetrics


def test_kalman_filter_initialization() -> None:
    kf = KalmanFilterTracker(dt=0.1)
    kf.initialize([100.0, 200.0])

    state = kf.get_state()
    assert state.shape == (4,)
    assert state[0] == 100.0
    assert state[1] == 200.0
    assert state[2] == 0.0
    assert state[3] == 0.0

    cov = kf.get_covariance()
    assert cov.shape == (4, 4)
    assert np.all(np.diag(cov) > 0)


def test_kalman_filter_predict_and_update() -> None:
    kf = KalmanFilterTracker(dt=1.0)
    # Set state with velocity: x=0, y=0, vx=10, vy=5
    kf.initialize([0.0, 0.0, 10.0, 5.0])

    # Predict: x should become 10, y should become 5
    pred = kf.predict()
    assert np.isclose(pred[0], 10.0)
    assert np.isclose(pred[1], 5.0)

    # Update with measurement at (11.0, 5.0)
    updated = kf.update([11.0, 5.0])
    pos = kf.get_position()
    assert 10.0 <= pos[0] <= 11.0
    assert np.isclose(pos[1], 5.0, atol=0.5)


def test_kalman_filter_occlusion_handling() -> None:
    kf = KalmanFilterTracker(dt=0.1)
    kf.initialize([50.0, 50.0])
    kf.predict()
    cov_before = kf.get_covariance()

    # Pass None as measurement (occluded)
    state = kf.update(None)
    assert kf.missed_detections_count == 1
    # State should remain predicted
    pos = kf.get_position()
    assert not np.isnan(pos[0]) and not np.isnan(pos[1])


def test_particle_filter_initialization() -> None:
    N = 500
    pf = ParticleFilterTracker(num_particles=N, dt=0.1)
    pf.initialize([150.0, 250.0], init_pos_std=1.0, init_vel_std=1.0)

    pos = pf.get_position()
    assert np.isclose(pos[0], 150.0, atol=2.0)
    assert np.isclose(pos[1], 250.0, atol=2.0)
    assert pf.num_particles == N
    assert np.isclose(np.sum(pf.weights), 1.0)
    assert pf.n_eff == float(N)


def test_particle_filter_vectorized_propagation() -> None:
    N = 200
    pf = ParticleFilterTracker(num_particles=N, dt=0.5, process_noise_pos=0.1, process_noise_vel=0.1)
    # Set deterministic particles with vx=20, vy=10
    pf.initialize([0.0, 0.0, 20.0, 10.0], init_pos_std=0.01, init_vel_std=0.01)

    pred = pf.predict()
    # After dt=0.5 with vx=20, vy=10, position should be around x=10, y=5
    assert np.isclose(pred[0], 10.0, atol=1.5)
    assert np.isclose(pred[1], 5.0, atol=1.5)


def test_particle_filter_resampling() -> None:
    N = 300
    pf = ParticleFilterTracker(num_particles=N, resample_threshold_ratio=0.5)
    pf.initialize([100.0, 100.0])

    # Skew weights to force N_eff < N/2
    pf.weights = np.zeros(N)
    pf.weights[0] = 0.9
    pf.weights[1:] = 0.1 / (N - 1)
    pf.n_eff = 1.0 / np.sum(pf.weights ** 2)
    assert pf.n_eff < pf.resample_threshold

    # Trigger update near target
    pf.update([100.0, 100.0])
    # After resampling, weights are reset to uniform 1/N
    assert np.isclose(pf.n_eff, float(N), atol=1.0)
    assert np.isclose(np.sum(pf.weights), 1.0)


def test_color_blob_detector_synthetic_frame() -> None:
    import cv2
    detector = ColorBlobDetector(color_preset="orange")

    # Create synthetic frame with orange circle
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (120, 140), 20, (0, 140, 255), -1)  # BGR Orange

    res = detector.detect(frame)
    assert res.is_detected
    assert res.centroid is not None
    assert np.isclose(res.centroid[0], 120.0, atol=3.0)
    assert np.isclose(res.centroid[1], 140.0, atol=3.0)


def test_tracking_metrics_computation() -> None:
    metrics = TrackingMetrics(recovery_threshold_px=10.0)

    # Frame 1: Perfect tracking
    metrics.record_frame(
        frame_index=1,
        timestamp_s=0.033,
        is_detected=True,
        is_occluded=False,
        gt_pos=(100.0, 100.0),
        meas_pos=(100.0, 100.0),
        kf_pos=(100.0, 100.0),
        kf_vel=(0.0, 0.0),
        pf_pos=(100.0, 100.0),
        pf_vel=(0.0, 0.0),
        kf_latency_ms=0.05,
        pf_latency_ms=0.30,
    )

    # Frame 2: Error 3 px for KF, 4 px for PF
    metrics.record_frame(
        frame_index=2,
        timestamp_s=0.066,
        is_detected=True,
        is_occluded=False,
        gt_pos=(100.0, 100.0),
        meas_pos=(100.0, 100.0),
        kf_pos=(103.0, 100.0),
        kf_vel=(0.0, 0.0),
        pf_pos=(100.0, 104.0),
        pf_vel=(0.0, 0.0),
        kf_latency_ms=0.04,
        pf_latency_ms=0.28,
    )

    summary = metrics.compute_summary()
    assert summary["total_frames"] == 2
    assert summary["visible_frames"] == 2
    assert summary["occluded_frames"] == 0
    assert summary["kalman_filter"]["overall"]["rmse"] > 0.0
    assert summary["particle_filter"]["overall"]["rmse"] > 0.0
    assert summary["kalman_filter"]["latency"]["mean_ms"] > 0.0


if __name__ == "__main__":
    test_kalman_filter_initialization()
    test_kalman_filter_predict_and_update()
    test_kalman_filter_occlusion_handling()
    test_particle_filter_initialization()
    test_particle_filter_vectorized_propagation()
    test_particle_filter_resampling()
    test_color_blob_detector_synthetic_frame()
    test_tracking_metrics_computation()
    print("All unit tests passed successfully!")
