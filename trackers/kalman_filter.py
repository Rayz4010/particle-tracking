"""
Linear Kalman Filter for 2D Kinematic Object Tracking.

Implements a discrete-time Linear Kalman Filter (KF) using a 2D Constant Velocity (CV)
motion model with state vector x = [x, y, vx, vy]^T.
All matrix algebra is implemented using NumPy from first principles.
"""

from typing import Optional, Tuple, Union
import numpy as np

from trackers.base_tracker import BaseTracker


class KalmanFilterTracker(BaseTracker):
    """
    Linear Kalman Filter Tracker.

    State:
        x = [x, y, vx, vy]^T (position and velocity in 2D Cartesian plane)
    Measurement:
        z = [z_x, z_y]^T (observed 2D centroid)
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 5.0,
        initial_error_std_pos: float = 10.0,
        initial_error_std_vel: float = 20.0,
    ) -> None:
        """
        Initialize Kalman Filter parameters.

        Args:
            dt: Default sample time interval (seconds).
            process_noise_std: Acceleration process noise standard deviation (sigma_a).
            measurement_noise_std: Measurement noise standard deviation (sigma_r).
            initial_error_std_pos: Initial position uncertainty standard deviation.
            initial_error_std_vel: Initial velocity uncertainty standard deviation.
        """
        super().__init__(name="Kalman Filter")
        self.dt: float = float(dt)
        self.process_noise_std: float = float(process_noise_std)
        self.measurement_noise_std: float = float(measurement_noise_std)
        self.initial_error_std_pos: float = float(initial_error_std_pos)
        self.initial_error_std_vel: float = float(initial_error_std_vel)

        # State vector: 4x1
        self.x: np.ndarray = np.zeros((4, 1), dtype=np.float64)

        # State error covariance: 4x4
        self.P: np.ndarray = np.eye(4, dtype=np.float64)

        # Measurement matrix H: 2x4 (maps state to 2D measurement)
        self.H: np.ndarray = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Measurement noise covariance R: 2x2
        self.R: np.ndarray = np.eye(2, dtype=np.float64) * (self.measurement_noise_std ** 2)

        # State transition matrix F: 4x4
        self.F: np.ndarray = self._construct_F(self.dt)

        # Process noise covariance Q: 4x4
        self.Q: np.ndarray = self._construct_Q(self.dt, self.process_noise_std)

        # Identity matrix for Joseph form
        self.I: np.ndarray = np.eye(4, dtype=np.float64)

    def _construct_F(self, dt: float) -> np.ndarray:
        """Construct state transition matrix for time step dt."""
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _construct_Q(self, dt: float, sigma_a: float) -> np.ndarray:
        """
        Construct Continuous White Noise Acceleration (CWNA) discrete Q matrix.
        Q = G * (sigma_a^2 * I_2) * G^T
        """
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        q_var = sigma_a ** 2

        Q = np.array([
            [dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0],
            [0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0],
            [dt3 / 2.0, 0.0,       dt2,       0.0],
            [0.0,       dt3 / 2.0, 0.0,       dt2]
        ], dtype=np.float64) * q_var
        return Q

    def initialize(
        self,
        initial_state: Union[np.ndarray, Tuple[float, float]],
        **kwargs
    ) -> None:
        """
        Initialize tracker state from initial measurement [x, y] or full state [x, y, vx, vy].
        """
        arr = np.asarray(initial_state, dtype=np.float64).flatten()
        if len(arr) == 2:
            self.x = np.array([[arr[0]], [arr[1]], [0.0], [0.0]], dtype=np.float64)
        elif len(arr) == 4:
            self.x = arr.reshape((4, 1))
        else:
            raise ValueError(f"Initial state must have length 2 or 4, got {len(arr)}")

        # Initialize error covariance P
        p_pos = self.initial_error_std_pos ** 2
        p_vel = self.initial_error_std_vel ** 2
        self.P = np.diag([p_pos, p_pos, p_vel, p_vel]).astype(np.float64)

        self.is_initialized = True
        self.missed_detections_count = 0
        self.total_frames_tracked = 0

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Execute Kalman Filter time propagation (prediction step).
        x^- = F * x
        P^- = F * P * F^T + Q
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker must be initialized before calling predict.")

        if dt is not None and abs(dt - self.dt) > 1e-6:
            self.dt = dt
            self.F = self._construct_F(dt)
            self.Q = self._construct_Q(dt, self.process_noise_std)

        # Prior state prediction
        self.x = self.F @ self.x

        # Prior covariance prediction
        self.P = (self.F @ self.P @ self.F.T) + self.Q

        self.total_frames_tracked += 1
        return self.x.flatten()

    def update(self, measurement: Optional[Union[np.ndarray, Tuple[float, float]]]) -> np.ndarray:
        """
        Execute Kalman Filter measurement update step.
        If measurement is None (occlusion/missed detection), runs predict-only continuation.
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker must be initialized before calling update.")

        if measurement is None:
            # Missed detection / occlusion: keep prior estimate
            self.missed_detections_count += 1
            return self.x.flatten()

        z_arr = np.asarray(measurement, dtype=np.float64).reshape((2, 1))
        if np.isnan(z_arr).any():
            self.missed_detections_count += 1
            return self.x.flatten()

        # Reset missed detection counter on valid measurement
        self.missed_detections_count = 0

        # Innovation (measurement residual)
        y = z_arr - (self.H @ self.x)

        # Innovation covariance
        S = (self.H @ self.P @ self.H.T) + self.R

        # Kalman gain: K = P * H^T * inv(S)
        # Solve S * K^T = H * P for numerical stability
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Posterior state update
        self.x = self.x + (K @ y)

        # Posterior covariance update using Joseph stabilized form:
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        IKH = self.I - (K @ self.H)
        self.P = (IKH @ self.P @ IKH.T) + (K @ self.R @ K.T)

        return self.x.flatten()

    def get_state(self) -> np.ndarray:
        """Return state vector [x, y, vx, vy]."""
        return self.x.flatten()

    def get_covariance(self) -> np.ndarray:
        """Return 4x4 state error covariance matrix P."""
        return self.P.copy()

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        super().reset()
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
