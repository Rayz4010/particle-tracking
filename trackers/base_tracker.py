"""
Abstract Base Tracker Interface for 2D Kinematic State Estimation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseTracker(ABC):
    """
    Abstract base class defining the tracking interface.
    
    All state estimation trackers adhere to the standard discrete
    predict-update lifecycle for 2D object tracking.
    State convention: [x, y, vx, vy]^T
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.is_initialized: bool = False
        self.missed_detections_count: int = 0
        self.total_frames_tracked: int = 0

    @abstractmethod
    def initialize(self, initial_state: np.ndarray, **kwargs) -> None:
        """
        Initialize the tracker state with an initial measurement or state vector.

        Args:
            initial_state: Array-like representing initial state [x, y, vx, vy]^T
                           or initial 2D measurement [x, y]^T.
            **kwargs: Additional tracker-specific parameters.
        """
        pass

    @abstractmethod
    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Advance tracker state by time interval dt using motion model.

        Args:
            dt: Time step in seconds. If None, default dt configured is used.

        Returns:
            np.ndarray: Predicted state vector [x, y, vx, vy]^T.
        """
        pass

    @abstractmethod
    def update(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        Correct the predicted state using current observation.
        If measurement is None (e.g., occlusion or detection dropout),
        the tracker maintains the predicted state and increments missed count.

        Args:
            measurement: 2D observation [z_x, z_y]^T, or None if missing.

        Returns:
            np.ndarray: Updated state vector [x, y, vx, vy]^T.
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return current state vector [x, y, vx, vy]^T."""
        pass

    def get_position(self) -> Tuple[float, float]:
        """Return current estimated 2D position (x, y)."""
        state = self.get_state()
        return float(state[0]), float(state[1])

    def get_velocity(self) -> Tuple[float, float]:
        """Return current estimated 2D velocity (vx, vy)."""
        state = self.get_state()
        return float(state[2]), float(state[3])

    @abstractmethod
    def get_covariance(self) -> Optional[np.ndarray]:
        """Return state error covariance matrix (4x4) or None if unavailable."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state to uninitialized."""
        self.is_initialized = False
        self.missed_detections_count = 0
        self.total_frames_tracked = 0
