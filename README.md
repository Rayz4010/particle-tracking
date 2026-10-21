# Real-Time Object Tracking Using Kalman and Particle Filter Comparison

A production-grade, modular Python application for real-time 2D single-object tracking and state estimation in live video streams (webcam feed, MP4 video, or synthetic trajectories). This suite provides a side-by-side empirical and theoretical comparison between two foundational state estimation algorithms implemented from first principles in NumPy:
1. **Linear Kalman Filter (KF)** with a 2D Constant Velocity (CV) motion model and Joseph-stabilized covariance updates.
2. **Sequential Importance Resampling (SIR) Particle Filter (PF)** with vectorized motion propagation, Gaussian likelihood evaluation, Effective Sample Size ($N_{eff}$) monitoring, and Systematic/Stratified Resampling.

---

## Architecture & System Design

```
project/
├── data/
│   ├── __init__.py
│   ├── synthetic_generator.py # Synthetic kinematics & video generator
│   ├── sample_video.mp4       # Pre-rendered 30 FPS test video
│   └── sample_ground_truth.csv# Mathematical Ground-Truth trajectory (x, y, vx, vy)
├── trackers/
│   ├── __init__.py
│   ├── base_tracker.py        # Abstract base class interface (typing & contracts)
│   ├── kalman_filter.py       # Linear Kalman Filter (matrix math from scratch)
│   └── particle_filter.py     # Vectorized SIR Particle Filter (NumPy broadcasting)
├── detectors/
│   ├── __init__.py
│   └── detector.py            # HSV blob extractor, ROI calibration, noise/occlusion injection
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # RMSE, MAE, latency profiler, occlusion recovery
│   └── visualizer.py          # OpenCV HUD, dual/split-screen, covariance ellipses, particles
├── tests/
│   └── test_trackers.py       # Unit test suite for kinematics and filters
├── main.py                    # Live camera/video player CLI with interactive HUD
├── benchmark.py               # Headless evaluation pipeline & Matplotlib chart generator
├── requirements.txt           # Dependencies
└── README.md                  # System manual and mathematical formulation
```

---

## Mathematical Formulation

Both algorithms track an object moving in a 2D Cartesian plane using the kinematic state vector:
$$\mathbf{x}_t = \begin{bmatrix} x_t \\ y_t \\ v_{x,t} \\ v_{y,t} \end{bmatrix} \in \mathbb{R}^4$$

Observations consist of 2D centroid measurements extracted by the computer vision detector:
$$\mathbf{z}_t = \begin{bmatrix} z_{x,t} \\ z_{y,t} \end{bmatrix} \in \mathbb{R}^2$$

---

### 1. Linear Kalman Filter (KF)

The Kalman filter is the optimal linear minimum mean-square error (MMSE) estimator under Gaussian assumptions.

#### Discrete-Time Motion Model
Assuming a constant velocity (CV) model across sample interval $\Delta t$:
$$\mathbf{x}_t = F \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q)$$

where the state transition matrix $F$ is:
$$F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

#### Process Noise Covariance $Q$
Modeled as Continuous White Noise Acceleration (CWNA) discretized over $\Delta t$ with acceleration variance $\sigma_a^2$:
$$Q = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} & 0 \\ 0 & \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} \\ \frac{\Delta t^3}{2} & 0 & \Delta t^2 & 0 \\ 0 & \frac{\Delta t^3}{2} & 0 & \Delta t^2 \end{bmatrix}$$

#### Measurement Model
$$\mathbf{z}_t = H \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, R)$$
$$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}, \quad R = \begin{bmatrix} \sigma_r^2 & 0 \\ 0 & \sigma_r^2 \end{bmatrix}$$

#### Predict Step
$$\hat{\mathbf{x}}_{t|t-1} = F \hat{\mathbf{x}}_{t-1|t-1}$$
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

#### Update Step (Handling Occlusion)
- **If measurement $\mathbf{z}_t$ is available:**
  - Measurement residual (innovation):
    $$\mathbf{y}_t = \mathbf{z}_t - H \hat{\mathbf{x}}_{t|t-1}$$
  - Innovation covariance:
    $$S_t = H P_{t|t-1} H^T + R$$
  - Kalman Gain:
    $$K_t = P_{t|t-1} H^T S_t^{-1}$$
  - Posterior state update:
    $$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t \mathbf{y}_t$$
  - Posterior error covariance (Joseph-stabilized formulation for positive definiteness):
    $$P_{t|t} = (I - K_t H) P_{t|t-1} (I - K_t H)^T + K_t R K_t^T$$
- **If measurement is missing (occlusion/missed detection):**
  - Execute predict-only step:
    $$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1}, \quad P_{t|t} = P_{t|t-1}$$

---

### 2. Sequential Importance Resampling (SIR) Particle Filter (PF)

The Particle Filter represents the posterior density using an empirical cloud of $N$ weighted particles $\{ X_t^{(i)}, w_t^{(i)} \}_{i=1}^N$.

#### Vectorized State Propagation
Each particle is propagated forward using the constant velocity model plus Gaussian stochastic diffusion:
$$X_t^{(i)} = X_{t-1}^{(i)} F^T + \mathcal{N}(0, \Sigma_{process})$$
Vectorized across all $N$ particles via NumPy broadcasting without Python loops.

#### Likelihood Weight Update
When observation $\mathbf{z}_t = [z_x, z_y]^T$ arrives, the likelihood of particle $i$ with position $(x^{(i)}, y^{(i)})$ is:
$$p(\mathbf{z}_t \mid X_t^{(i)}) = \frac{1}{2\pi \sigma_{meas}^2} \exp\left( - \frac{(x^{(i)} - z_x)^2 + (y^{(i)} - z_y)^2}{2 \sigma_{meas}^2} \right) + \epsilon$$

Weights are updated and normalized:
$$\tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot p(\mathbf{z}_t \mid X_t^{(i)}), \quad w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^N \tilde{w}_t^{(j)}}$$
*Under occlusion, weights remain unchanged and particles continue stochastic motion propagation.*

#### Degeneracy Monitoring & Resampling
Degeneracy is monitored via the Effective Sample Size:
$$N_{eff} = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2}$$
When $N_{eff} < \frac{N}{2}$, resampling is triggered.

- **Systematic Resampling:**
  Generates $N$ ordered sampling positions $u_k = \frac{k - 1 + u_0}{N}$ where $u_0 \sim \mathcal{U}[0, 1)$.
  Particles are sampled in $\mathcal{O}(N)$ vectorized time using `np.searchsorted(np.cumsum(w), u)`.
- **Particle Roughening / Jitter:**
  A small Gaussian perturbation is injected post-resampling to combat sample impoverishment.
  Weights are reset to $w_t^{(i)} = \frac{1}{N}$.

#### State and Covariance Estimation
$$\hat{\mathbf{x}}_t = \sum_{i=1}^N w_t^{(i)} X_t^{(i)}, \quad \hat{P}_t = \sum_{i=1}^N w_t^{(i)} (X_t^{(i)} - \hat{\mathbf{x}}_t)(X_t^{(i)} - \hat{\mathbf{x}}_t)^T$$

---

## Installation & Requirements

- Python 3.10+
- Dependencies: `opencv-python`, `numpy`, `scipy`, `matplotlib`

```bash
# Clone or navigate to the repository
cd realtime_object_tracking

# Install dependencies into your Python environment
pip install -r requirements.txt
```

*(If using Anaconda: activate your conda environment, e.g. `conda activate tf2.18`)*

---

## Quickstart & Usage

### 1. Interactive Web Dashboard (`app.py` & `main.py --web`)

Launch the full-featured browser dashboard powered by Streamlit and Plotly:
```bash
streamlit run app.py
# Or via main.py:
python main.py --web
```

#### Web Dashboard Highlights:
- **🎥 Live Tracking Studio**: Interactive parameter adjustment (particle count, noise, periodic & manual occlusion), real-time annotated video player (Overlay or Split-Screen), live KPI metric cards, and rolling error time-series plots.
- **📊 Benchmarking Lab**: One-click headless benchmark executor across multiple kinematic scenarios, comparative tables, interactive plots (`rmse_comparison.png`, `latency_boxplot.png`, `occlusion_recovery.png`), and CSV download.
- **📖 Algorithm Formulations**: Side-by-side mathematical formulation explorer and complexity comparison.

---

### 2. Interactive Desktop Video Player (`main.py`)

Run live tracking on the bundled test video:
```bash
python main.py --source data/sample_video.mp4
```

Run on a live webcam feed:
```bash
python main.py --source 0 --select-roi
```

Run directly on a real-time synthetic trajectory stream:
```bash
python main.py --source synthetic --particles 1000 --mode overlay
```

#### Interactive Keyboard Controls
| Key | Action | Description |
|---|---|---|
| `q` / `ESC` | **Quit** | Exits the application and displays telemetry summary |
| `s` | **Select ROI** | Pauses feed and allows clicking & dragging a box to auto-calibrate HSV color |
| `o` | **Toggle Occlusion** | Manually suppresses detector measurement to evaluate tracker dead reckoning |
| `p` | **Pause / Resume** | Toggles video playback pause |
| `m` | **Toggle View Mode** | Switches between Combined Overlay and Side-by-Side Split Screen |
| `c` | **Clear Trails** | Clears trajectory history lines on screen |
| `r` | **Reset Filters** | Clears filter states and re-acquires the object on next detection |

#### Additional CLI Options
```bash
python main.py \
  --source data/sample_video.mp4 \
  --color orange \
  --particles 1000 \
  --noise 3.0 \
  --occlusion-freq 90 \
  --occlusion-duration 25 \
  --mode split \
  --save-video output.mp4 \
  --save-csv telemetry.csv
```

---

### 2. Automated Offline Benchmark Suite (`benchmark.py`)

Executes headless evaluation against ground truth, exports detailed telemetry to CSV, and generates publication-grade Matplotlib comparison charts:

```bash
python benchmark.py --frames 300 --particles 1000 --noise 3.0 --output-dir benchmark_output
```

To run benchmarks on different synthetic kinematic scenarios:
```bash
# Linear constant-velocity trajectory with wall collisions
python benchmark.py --video synthetic --trajectory linear --frames 250 --output-dir benchmark_output/linear

# Harmonic oscillation / curved trajectory (stress-testing constant-velocity assumption)
python benchmark.py --video synthetic --trajectory mixed --frames 300 --output-dir benchmark_output/mixed
```

#### Generated Artifacts
1. **`rmse_comparison.png`**:
   - 2D spatial trajectory comparison (Ground Truth vs. Kalman Filter vs. Particle Filter).
   - Time-series tracking error (pixels) with shaded occlusion intervals.
2. **`latency_boxplot.png`**:
   - Box plot and histogram density comparison of per-frame execution latency (ms).
3. **`occlusion_recovery.png`**:
   - Zoomed-in transient response showing tracking error during target occlusion and recovery dynamics upon reappearance.
4. **`benchmark_results.csv`**:
   - Full frame-by-frame telemetry log (timestamps, GT positions, measurements, filter states, errors, latencies, $N_{eff}$).

---

### 3. Synthetic Data Generator (`data/synthetic_generator.py`)

Generate custom test videos and mathematical ground-truth CSV trajectories:
```bash
python data/synthetic_generator.py
```

---

### 4. Running Unit Tests

Run the test suite covering matrix dimensions, constant velocity propagation, Joseph-form covariance updates, vectorized resampling, and metric computations:
```bash
python tests/test_trackers.py
```

---

## Empirical Benchmark Results

Evaluated on 300 frames of non-linear mixed trajectory with an occlusion wall and $\sigma_{noise} = 3.0\text{ px}$:

| Metric | Linear Kalman Filter (KF) | Particle Filter (PF, $N=1000$) |
|---|:---:|:---:|
| **Visible-Only RMSE** | 76.81 px | **4.65 px** |
| **Overall RMSE** | 81.87 px | **9.32 px** |
| **Overall MAE** | 73.24 px | **5.74 px** |
| **Occlusion Peak Error** | 193.41 px | **59.69 px** |
| **Mean Latency per Frame** | **0.038 ms** | 0.256 ms |
| **Median Latency per Frame** | **0.029 ms** | 0.215 ms |
| **P95 Latency** | **0.076 ms** | 0.357 ms |
| **Theoretical Algorithmic Throughput**| **~26,000 FPS** | **~3,900 FPS** |
| **Occlusion Recovery** | 4.0 frames | **0.0 frames** (instant re-acquisition) |

### Key Insights & Trade-Offs
1. **Curved Kinematics & Non-Linearity:** When an object follows a curving trajectory, the linear constant-velocity Kalman Filter lags behind the true acceleration unless process noise is tuned very high (which increases susceptibility to measurement noise). The Particle Filter effortlessly tracks curved paths and multimodal distributions through non-linear particle swarms and resampling.
2. **Computational Complexity:** The Kalman Filter executes closed-form matrix math in $\mathcal{O}(d^3)$ where $d=4$ is state dimension (~$0.03\text{ ms}$), providing unmatched speed. The Particle Filter with $N=1000$ particles executes in $\mathcal{O}(N \cdot d)$ time (~$0.25\text{ ms}$), achieving over 3,900 FPS thanks to NumPy vectorization, easily exceeding real-time video requirements (30–60 FPS).
3. **Occlusion Robustness:** Both filters successfully coast (predict-only) during occlusions. Upon target reappearance, the Particle Filter cloud re-seeds immediately around the newly detected centroid, demonstrating instantaneous re-acquisition.
