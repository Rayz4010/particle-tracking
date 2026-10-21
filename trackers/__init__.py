"""
Trackers module implementing state estimation algorithms:
- BaseTracker: Abstract base class for 2D kinematic trackers
- KalmanFilterTracker: Linear Kalman Filter with Constant Velocity model
- ParticleFilterTracker: Sequential Importance Resampling (SIR) Particle Filter
"""

from trackers.base_tracker import BaseTracker
from trackers.kalman_filter import KalmanFilterTracker
from trackers.particle_filter import ParticleFilterTracker

__all__ = ["BaseTracker", "KalmanFilterTracker", "ParticleFilterTracker"]
