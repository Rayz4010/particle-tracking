"""
Sequential Importance Resampling (SIR) Particle Filter for 2D Object Tracking.

Implements a vectorized Bootstrap/SIR Particle Filter with:
- State representation per particle: [x, y, vx, vy]^T
- Vectorized Constant Velocity motion model propagation via NumPy broadcasting
- Gaussian likelihood weight updating
- Effective Sample Size (N_eff) monitoring
- Systematic and Stratified Resampling algorithms
- Weighted empirical state mean and covariance estimation
"""

from typing import Optional, Tuple, Union, Literal
import numpy as np

from trackers.base_tracker import BaseTracker


class ParticleFilterTracker(BaseTracker):
    """
    Sampling Importance Resampling (SIR) Particle Filter Tracker.
    
    Tracks 2D kinematics using N weighted particles:
        X in R^{N x 4}: each particle is [x, y, vx, vy]
        w in R^N: normalized particle weights
    """

    def __init__(
        self,
        num_particles: int = 1000,
        dt: float = 1.0 / 30.0,
        process_noise_pos: float = 3.0,
        process_noise_vel: float = 6.0,
        measurement_noise_std: float = 15.0,
        resample_threshold_ratio: float = 0.5,
        resample_method: Literal["systematic", "stratified"] = "systematic",
        roughening_std: float = 0.5,
    ) -> None:
        """
        Initialize Particle Filter parameters.

        Args:
            num_particles: Number of particles (N), typically 500-1000.
            dt: Time step duration in seconds.
            process_noise_pos: Process noise standard deviation for position (pixels).
            process_noise_vel: Process noise standard deviation for velocity (pixels/s).
            measurement_noise_std: Measurement likelihood standard deviation (pixels).
            resample_threshold_ratio: Resample trigger threshold (N_eff < ratio * N).
            resample_method: 'systematic' or 'stratified'.
            roughening_std: Small jitter added post-resampling to avoid impoverishment.
        """
        super().__init__(name="Particle Filter")
        self.num_particles: int = int(num_particles)
        self.dt: float = float(dt)
        self.process_noise_pos: float = float(process_noise_pos)
        self.process_noise_vel: float = float(process_noise_vel)
        self.measurement_noise_std: float = float(measurement_noise_std)
        self.resample_threshold_ratio: float = float(resample_threshold_ratio)
        self.resample_threshold: float = self.num_particles * self.resample_threshold_ratio
        self.resample_method: str = resample_method
        self.roughening_std: float = float(roughening_std)

        # Particle states: (N, 4)
        self.particles: np.ndarray = np.zeros((self.num_particles, 4), dtype=np.float64)

        # Particle weights: (N,), normalized
        self.weights: np.ndarray = np.ones(self.num_particles, dtype=np.float64) / self.num_particles

        # Effective sample size
        self.n_eff: float = float(self.num_particles)

        # Estimated state and covariance
        self.state_estimate: np.ndarray = np.zeros(4, dtype=np.float64)
        self.covariance_estimate: np.ndarray = np.eye(4, dtype=np.float64)

        # Precompute transition matrix F: 4x4
        self.F: np.ndarray = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0,     self.dt],
            [0.0, 0.0, 1.0,     0.0],
            [0.0, 0.0, 0.0,     1.0]
        ], dtype=np.float64)

    def initialize(
        self,
        initial_state: Union[np.ndarray, Tuple[float, float]],
        init_pos_std: float = 15.0,
        init_vel_std: float = 10.0,
        **kwargs
    ) -> None:
        """
        Distribute particles around initial measurement or state.

        Args:
            initial_state: [x, y] or [x, y, vx, vy].
            init_pos_std: Initial spread standard deviation for position.
            init_vel_std: Initial spread standard deviation for velocity.
        """
        arr = np.asarray(initial_state, dtype=np.float64).flatten()
        if len(arr) == 2:
            x0, y0 = arr[0], arr[1]
            vx0, vy0 = 0.0, 0.0
        elif len(arr) == 4:
            x0, y0, vx0, vy0 = arr[0], arr[1], arr[2], arr[3]
        else:
            raise ValueError(f"Initial state must have length 2 or 4, got {len(arr)}")

        # Sample initial particle swarm from Gaussian distribution
        self.particles[:, 0] = np.random.normal(x0, init_pos_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(y0, init_pos_std, self.num_particles)
        self.particles[:, 2] = np.random.normal(vx0, init_vel_std, self.num_particles)
        self.particles[:, 3] = np.random.normal(vy0, init_vel_std, self.num_particles)

        # Uniform initial weights
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.n_eff = float(self.num_particles)

        self._compute_state_estimates()

        self.is_initialized = True
        self.missed_detections_count = 0
        self.total_frames_tracked = 0

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Propagate particles forward in time using constant velocity motion model
        and vectorized Gaussian process noise.

        X_t = X_{t-1} @ F^T + Noise
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker must be initialized before calling predict.")

        if dt is not None and abs(dt - self.dt) > 1e-6:
            self.dt = dt
            self.F = np.array([
                [1.0, 0.0, dt,  0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)

        # Vectorized deterministic propagation: (N, 4) @ (4, 4)^T -> (N, 4)
        self.particles = self.particles @ self.F.T

        # Add stochastic process noise
        pos_noise = np.random.normal(0.0, self.process_noise_pos, size=(self.num_particles, 2))
        vel_noise = np.random.normal(0.0, self.process_noise_vel, size=(self.num_particles, 2))
        self.particles[:, 0:2] += pos_noise
        self.particles[:, 2:4] += vel_noise

        self._compute_state_estimates()
        self.total_frames_tracked += 1
        return self.state_estimate

    def update(self, measurement: Optional[Union[np.ndarray, Tuple[float, float]]]) -> np.ndarray:
        """
        Measurement likelihood weighting and resampling.
        If measurement is None (occlusion), particles propagate without likelihood penalty.
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker must be initialized before calling update.")

        if measurement is None:
            # Occlusion / missed detection
            self.missed_detections_count += 1
            # Recompute state estimate from predicted particles
            self._compute_state_estimates()
            return self.state_estimate

        z = np.asarray(measurement, dtype=np.float64).flatten()
        if len(z) < 2 or np.isnan(z[0]) or np.isnan(z[1]):
            self.missed_detections_count += 1
            self._compute_state_estimates()
            return self.state_estimate

        self.missed_detections_count = 0
        zx, zy = z[0], z[1]

        # Vectorized Euclidean squared distance: (N,)
        dx = self.particles[:, 0] - zx
        dy = self.particles[:, 1] - zy
        sq_dist = (dx * dx) + (dy * dy)

        # Gaussian likelihood calculation: p(z | x_i)
        # Adding tiny epsilon to prevent zero likelihood
        var = self.measurement_noise_std ** 2
        likelihood = np.exp(-0.5 * sq_dist / var) + 1e-12

        # Update unnormalized weights
        self.weights *= likelihood

        # Normalize weights
        sum_weights = np.sum(self.weights)
        if sum_weights > 0.0 and np.isfinite(sum_weights):
            self.weights /= sum_weights
        else:
            # Degeneracy recovery: reset to uniform weights
            self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles

        # Calculate Effective Sample Size (N_eff = 1 / sum(w^2))
        sum_sq = np.sum(self.weights ** 2)
        self.n_eff = 1.0 / max(sum_sq, 1e-12)

        # Trigger Resampling if N_eff falls below threshold
        if self.n_eff < self.resample_threshold:
            self._resample()

        self._compute_state_estimates()
        return self.state_estimate

    def _resample(self) -> None:
        """
        Execute Systematic or Stratified Resampling in O(N) vectorized time.
        """
        if self.resample_method == "stratified":
            indices = self._stratified_resample(self.weights)
        else:
            indices = self._systematic_resample(self.weights)

        # Duplicate/prune particles according to sampled indices
        self.particles = self.particles[indices]

        # Apply small jitter (roughening) to prevent sample impoverishment
        if self.roughening_std > 0.0:
            jitter = np.random.normal(0.0, self.roughening_std, size=self.particles.shape)
            self.particles += jitter

        # Reset weights to uniform after resampling
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.n_eff = float(self.num_particles)

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Systematic Resampling:
        Generates N points on a deterministic grid u_k = (k + u0) / N with u0 ~ U(0, 1).
        Vectorized with np.searchsorted.
        """
        n = self.num_particles
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Avoid floating point ceiling issues

        u0 = np.random.uniform(0.0, 1.0 / n)
        u = u0 + np.arange(n, dtype=np.float64) / n
        return np.searchsorted(cumsum, u)

    def _stratified_resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Stratified Resampling:
        Partitions [0, 1) into N equal bins, draws u_k ~ U(k/N, (k+1)/N).
        Vectorized with np.searchsorted.
        """
        n = self.num_particles
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0

        u = (np.arange(n, dtype=np.float64) + np.random.uniform(0.0, 1.0, size=n)) / n
        return np.searchsorted(cumsum, u)

    def _compute_state_estimates(self) -> None:
        """
        Compute weighted empirical mean and covariance from particle swarm.
        """
        # Weighted mean: shape (4,)
        self.state_estimate = np.average(self.particles, axis=0, weights=self.weights)

        # Weighted covariance: shape (4, 4)
        diff = self.particles - self.state_estimate
        # (diff.T * weights) @ diff
        self.covariance_estimate = (diff.T * self.weights) @ diff

    def get_state(self) -> np.ndarray:
        """Return state vector estimate [x, y, vx, vy]."""
        return self.state_estimate.copy()

    def get_covariance(self) -> np.ndarray:
        """Return 4x4 estimated covariance matrix."""
        return self.covariance_estimate.copy()

    def get_particles(self, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve particle coordinates and weights, optionally downsampled for rendering.

        Returns:
            Tuple of (particles[:, :2], weights)
        """
        if sample_size is not None and sample_size < self.num_particles:
            indices = np.random.choice(self.num_particles, size=sample_size, replace=False)
            return self.particles[indices, :2], self.weights[indices]
        return self.particles[:, :2], self.weights

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        super().reset()
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float64)
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.n_eff = float(self.num_particles)
        self.state_estimate = np.zeros(4, dtype=np.float64)
        self.covariance_estimate = np.eye(4, dtype=np.float64)
