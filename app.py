"""
Streamlit Web Frontend for Real-Time Object Tracking:
Kalman Filter vs. Particle Filter Comparison.

Features:
- Live Interactive Tracking Playground with synthetic streams, sample videos, or custom uploads
- Dynamic parameter tuning (particle count, measurement noise, occlusions, display modes)
- Real-time telemetry KPIs and rolling error Plotly charts
- Automated Benchmarking Lab with interactive metrics, publication plots, and CSV export
- Algorithmic mathematical deep-dive and theoretical formulation guide
"""

from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.synthetic_generator import SyntheticStream, create_sample_video
from detectors.detector import ColorBlobDetector, DetectionResult, COLOR_PRESETS
from trackers.kalman_filter import KalmanFilterTracker
from trackers.particle_filter import ParticleFilterTracker
from utils.metrics import FrameRecord, TrackingMetrics
from utils.visualizer import RenderMode, TrackingVisualizer


# Streamlit Page Configuration
st.set_page_config(
    page_title="Object Tracking: Kalman vs Particle Filter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.image("https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png", width=64)
    st.sidebar.title("Configuration")

    nav_tab = st.sidebar.radio(
        "Navigation",
        ["🎥 Live Tracking Studio", "📊 Benchmarking Lab", "📖 Algorithm Formulations"],
        index=0,
    )
    st.sidebar.markdown("---")
    return nav_tab


# =========================================================================
# TAB 1: Live Tracking Studio
# =========================================================================
def render_live_tracking_studio():
    st.markdown('<div class="main-title">🎯 Live Tracking Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real-time side-by-side state estimation comparison: '
        'Linear Kalman Filter (KF) vs. Sequential Importance Resampling Particle Filter (PF)</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_display = st.columns([1, 2.5])

    with col_ctrl:
        st.subheader("⚙️ Stream Controls")

        source_type = st.selectbox(
            "Video Source",
            ["Synthetic Stream (Mixed Trajectory)",
             "Synthetic Stream (Linear Bounce)",
             "Synthetic Stream (Circular Path)",
             "Pre-rendered Sample Video (MP4)",
             "Upload Video File"],
            index=0,
        )

        uploaded_file = None
        if source_type == "Upload Video File":
            uploaded_file = st.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])

        st.markdown("#### Tracker Parameters")
        num_particles = st.slider("Particles (N)", min_value=100, max_value=2500, value=1000, step=100)
        resample_method = st.selectbox("PF Resampling Algorithm", ["systematic", "stratified"], index=0)

        st.markdown("#### Noise & Occlusion Stress Test")
        meas_noise = st.slider("Synthetic Measurement Noise (px)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
        manual_occlusion = st.checkbox("Simulate Immediate Occlusion (Dropout)", value=False)
        periodic_occ = st.checkbox("Simulate Periodic Occlusion", value=False)
        occ_interval = 90
        occ_duration = 30
        if periodic_occ:
            occ_interval = st.slider("Occlusion Period (Frames)", min_value=30, max_value=150, value=90, step=10)
            occ_duration = st.slider("Occlusion Duration (Frames)", min_value=10, max_value=60, value=30, step=5)

        st.markdown("#### Visualization Display")
        vis_mode = st.radio("Display Mode", ["Overlay", "Split-Screen"], index=0)
        max_frames_to_run = st.slider("Frames to Stream", min_value=50, max_value=600, value=250, step=50)

        start_btn = st.button("▶️ Run Tracking Stream", type="primary", use_container_width=True)

    with col_display:
        st.subheader("📺 Visualizer View & Telemetry")
        frame_placeholder = st.empty()

        # Telemetry metrics cards placeholder
        kpi_cols = st.columns(4)
        kpi_kf_err = kpi_cols[0].empty()
        kpi_pf_err = kpi_cols[1].empty()
        kpi_kf_lat = kpi_cols[2].empty()
        kpi_pf_lat = kpi_cols[3].empty()

        chart_placeholder = st.empty()

    if start_btn:
        # Determine source
        using_synthetic = False
        synth_stream = None
        cap = None
        fps = 30.0
        dt = 1.0 / fps

        if "Synthetic" in source_type:
            using_synthetic = True
            traj = "mixed"
            if "Linear" in source_type:
                traj = "linear"
            elif "Circular" in source_type:
                traj = "circular"

            synth_stream = SyntheticStream(
                width=800,
                height=600,
                fps=fps,
                trajectory=traj,
                add_occlusion_barrier=True,
            )
        elif source_type == "Pre-rendered Sample Video (MP4)":
            sample_vid_path = Path("data/sample_video.mp4")
            if not sample_vid_path.exists():
                with st.spinner("Generating sample benchmark video..."):
                    create_sample_video()
            cap = cv2.VideoCapture(str(sample_vid_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        elif source_type == "Upload Video File":
            if uploaded_file is None:
                st.error("Please upload a video file to proceed.")
                return
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Instantiate components
        detector = ColorBlobDetector(
            color_preset="orange",
            measurement_noise_std=meas_noise,
            periodic_occlusion_interval=occ_interval if periodic_occ else 0,
            periodic_occlusion_duration=occ_duration if periodic_occ else 0,
        )
        detector.is_manual_occlusion = manual_occlusion

        render_m = RenderMode.SPLIT_SCREEN if vis_mode == "Split-Screen" else RenderMode.OVERLAY
        visualizer = TrackingVisualizer(trail_length=60, render_mode=render_m)

        kf = KalmanFilterTracker(
            dt=dt,
            process_noise_std=8.0,
            measurement_noise_std=max(3.0, meas_noise),
        )
        pf = ParticleFilterTracker(
            num_particles=num_particles,
            dt=dt,
            process_noise_pos=4.0,
            process_noise_vel=25.0,
            measurement_noise_std=max(12.0, meas_noise * 2.0),
            resample_threshold_ratio=0.5,
            resample_method=resample_method,
        )

        metrics = TrackingMetrics(recovery_threshold_px=15.0)

        # Telemetry histories for live plotting
        hist_frames = []
        hist_kf_err = []
        hist_pf_err = []

        trackers_initialized = False
        frame_idx = 0

        progress_bar = st.progress(0)

        while frame_idx < max_frames_to_run:
            frame_idx += 1
            gt_pos = None
            gt_vel = None
            sim_occ = False

            if using_synthetic:
                assert synth_stream is not None
                ret, frame, s_gt_pos, s_gt_vel, s_occ = synth_stream.read()
                if not ret:
                    break
                gt_pos = s_gt_pos
                gt_vel = s_gt_vel
                sim_occ = s_occ
            else:
                assert cap is not None
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            # 1. Detection
            t_det0 = time.perf_counter()
            det_res = detector.detect(frame)
            t_det_ms = (time.perf_counter() - t_det0) * 1000.0

            if sim_occ:
                det_res.is_occluded = True
                det_res.centroid = None

            is_lost = det_res.is_occluded or (not det_res.is_detected)
            meas = None if is_lost else det_res.centroid

            # Initialize upon first unoccluded detection
            if not trackers_initialized:
                init_pt = meas if meas is not None else gt_pos
                if init_pt is not None:
                    kf.initialize(init_pt)
                    pf.initialize(init_pt, init_pos_std=10.0, init_vel_std=60.0)
                    trackers_initialized = True
                continue

            # 2. Kalman Filter
            t_kf0 = time.perf_counter()
            kf.predict(dt=dt)
            kf.update(meas)
            t_kf_ms = (time.perf_counter() - t_kf0) * 1000.0

            # 3. Particle Filter
            t_pf0 = time.perf_counter()
            pf.predict(dt=dt)
            pf.update(meas)
            t_pf_ms = (time.perf_counter() - t_pf0) * 1000.0

            # Telemetry record
            rec = metrics.record_frame(
                frame_index=frame_idx,
                timestamp_s=frame_idx * dt,
                is_detected=det_res.is_detected,
                is_occluded=det_res.is_occluded,
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

            hist_frames.append(frame_idx)
            hist_kf_err.append(rec.kf_error)
            hist_pf_err.append(rec.pf_error)

            # Render annotated frame
            vis_frame = visualizer.render_frame(
                frame=frame,
                detection=det_res,
                kf_state=kf.get_state(),
                kf_cov=kf.get_covariance(),
                pf_state=pf.get_state(),
                pf_particles=pf.get_particles(sample_size=250),
                fps=fps,
                record=rec,
                is_manual_occlusion=detector.is_manual_occlusion,
                spatial_box=detector.spatial_occlusion_box,
            )

            # Convert BGR to RGB for Streamlit rendering
            vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)

            # Update KPI cards
            kpi_kf_err.metric("Kalman Error", f"{rec.kf_error:.1f} px", delta=None)
            kpi_pf_err.metric("Particle Error", f"{rec.pf_error:.1f} px", delta=None)
            kpi_kf_lat.metric("KF Latency", f"{rec.kf_latency_ms:.3f} ms")
            kpi_pf_lat.metric("PF Latency", f"{rec.pf_latency_ms:.3f} ms")

            progress_bar.progress(min(1.0, frame_idx / max_frames_to_run))

        if cap is not None:
            cap.release()

        # Render complete session plot
        if hist_frames:
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(x=hist_frames, y=hist_kf_err, mode="lines", name="Kalman Filter Error (px)", line=dict(color="#0072B2", width=2)))
            fig_err.add_trace(go.Scatter(x=hist_frames, y=hist_pf_err, mode="lines", name="Particle Filter Error (px)", line=dict(color="#D55E00", width=2)))
            fig_err.update_layout(
                title="Tracking Error Trajectory Over Time (px)",
                xaxis_title="Frame Index",
                yaxis_title="Euclidean Error (pixels)",
                template="plotly_white",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            chart_placeholder.plotly_chart(fig_err, use_container_width=True)


# =========================================================================
# TAB 2: Benchmarking Lab
# =========================================================================
def render_benchmarking_lab():
    st.markdown('<div class="main-title">📊 Evaluation & Benchmarking Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Headless empirical evaluation across kinematic scenarios, '
        'RMSE estimation error, latency distributions, and occlusion recovery metrics.</div>',
        unsafe_allow_html=True,
    )

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        bench_scenario = st.selectbox(
            "Evaluation Scenario",
            ["mixed (Harmonic & Curved)", "linear (Constant Velocity)", "circular (Curvature Test)"],
            index=0,
        )
    with b_col2:
        bench_frames = st.slider("Benchmark Frames", min_value=100, max_value=500, value=300, step=50)
    with b_col3:
        bench_particles = st.slider("Particles Count", min_value=200, max_value=2000, value=1000, step=200)
    with b_col4:
        bench_noise = st.slider("Measurement Noise (px)", min_value=0.0, max_value=10.0, value=3.0, step=1.0)

    run_bench_btn = st.button("🚀 Execute Comprehensive Benchmark", type="primary", use_container_width=True)

    traj_key = bench_scenario.split()[0]
    out_dir = Path(f"benchmark_output/{traj_key}")

    if run_bench_btn:
        with st.spinner("Executing headless benchmark simulation..."):
            from benchmark import run_benchmark
            metrics = run_benchmark(
                video_source="synthetic",
                num_frames=bench_frames,
                num_particles=bench_particles,
                synthetic_noise_std=bench_noise,
                trajectory_type=traj_key,
                output_dir=str(out_dir),
            )
            st.success("Benchmark completed successfully! Artifacts generated.")

    # Check if benchmark artifacts exist
    csv_file = out_dir / "benchmark_results.csv"
    if not csv_file.exists():
        # Fallback to default output
        out_dir = Path("benchmark_output")
        csv_file = out_dir / "benchmark_results.csv"

    if csv_file.exists():
        df_bench = pd.read_csv(csv_file)

        st.markdown("### 🏆 Benchmark Summary Table")
        # Compute summary values
        kf_rmse = np.sqrt(np.mean(df_bench["kf_error_px"] ** 2))
        pf_rmse = np.sqrt(np.mean(df_bench["pf_error_px"] ** 2))
        kf_mae = np.mean(df_bench["kf_error_px"])
        pf_mae = np.mean(df_bench["pf_error_px"])
        kf_lat_mean = np.mean(df_bench["kf_latency_ms"])
        pf_lat_mean = np.mean(df_bench["pf_latency_ms"])
        kf_fps = 1000.0 / max(kf_lat_mean, 1e-4)
        pf_fps = 1000.0 / max(pf_lat_mean, 1e-4)

        summary_data = {
            "Metric": [
                "Overall RMSE (pixels)",
                "Overall MAE (pixels)",
                "Peak Maximum Error (pixels)",
                "Mean Execution Latency (ms)",
                "Median Execution Latency (ms)",
                "P95 Execution Latency (ms)",
                "Theoretical Throughput (FPS)",
            ],
            "Linear Kalman Filter (KF)": [
                f"{kf_rmse:.2f} px",
                f"{kf_mae:.2f} px",
                f"{df_bench['kf_error_px'].max():.2f} px",
                f"{kf_lat_mean:.4f} ms",
                f"{df_bench['kf_latency_ms'].median():.4f} ms",
                f"{np.percentile(df_bench['kf_latency_ms'], 95):.4f} ms",
                f"{kf_fps:,.1f} FPS",
            ],
            "SIR Particle Filter (PF)": [
                f"{pf_rmse:.2f} px",
                f"{pf_mae:.2f} px",
                f"{df_bench['pf_error_px'].max():.2f} px",
                f"{pf_lat_mean:.4f} ms",
                f"{df_bench['pf_latency_ms'].median():.4f} ms",
                f"{np.percentile(df_bench['pf_latency_ms'], 95):.4f} ms",
                f"{pf_fps:,.1f} FPS",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        st.markdown("### 📈 Comparative Visualizations")
        tab_img1, tab_img2, tab_img3, tab_raw = st.tabs([
            "📉 Trajectory & RMSE Error",
            "⏱️ Execution Latency",
            "🛡️ Occlusion Dynamics",
            "📋 Raw Telemetry Data"
        ])

        with tab_img1:
            img_rmse = out_dir / "rmse_comparison.png"
            if img_rmse.exists():
                st.image(str(img_rmse), caption="2D Trajectory Estimation & Error Time-Series with Occlusion Intervals", use_container_width=True)
            else:
                st.info("Run benchmark to generate RMSE comparison plot.")

        with tab_img2:
            img_lat = out_dir / "latency_boxplot.png"
            if img_lat.exists():
                st.image(str(img_lat), caption="Execution Latency Distribution Boxplot & Density Histograms", use_container_width=True)
            else:
                st.info("Run benchmark to generate Latency boxplot.")

        with tab_img3:
            img_occ = out_dir / "occlusion_recovery.png"
            if img_occ.exists():
                st.image(str(img_occ), caption="Transient Error Dynamics During and Post-Occlusion Barrier", use_container_width=True)
            else:
                st.info("Run benchmark to generate Occlusion recovery plot.")

        with tab_raw:
            st.dataframe(df_bench, use_container_width=True)
            csv_bytes = df_bench.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Telemetry CSV",
                data=csv_bytes,
                file_name="benchmark_results.csv",
                mime="text/csv",
                type="primary",
            )
    else:
        st.info("No benchmark results found yet. Click 'Execute Comprehensive Benchmark' above to begin.")


# =========================================================================
# TAB 3: Algorithm Formulations
# =========================================================================
def render_algorithm_formulations():
    st.markdown('<div class="main-title">📖 State Estimation Theory & Formulations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Rigorous mathematical comparison of the Discrete Linear Kalman Filter '
        'and the Sequential Importance Resampling (SIR) Particle Filter.</div>',
        unsafe_allow_html=True,
    )

    col_kf, col_pf = st.columns(2)

    with col_kf:
        st.markdown("""
        ### 🔵 Linear Kalman Filter (KF)
        The optimal Linear Quadratic Gaussian state estimator for continuous/discrete linear dynamical systems.

        #### 1. Kinematic State Space
        $$\\mathbf{x} = \\begin{bmatrix} x & y & v_x & v_y \\end{bmatrix}^T \\in \\mathbb{R}^4$$

        #### 2. Transition Matrix (Constant Velocity)
        $$F(\\Delta t) = \\begin{bmatrix} 1 & 0 & \\Delta t & 0 \\\\ 0 & 1 & 0 & \\Delta t \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}$$

        #### 3. Continuous White Noise Acceleration $Q$
        $$Q = \\sigma_a^2 \\begin{bmatrix} \\frac{\\Delta t^4}{4} & 0 & \\frac{\\Delta t^3}{2} & 0 \\\\ 0 & \\frac{\\Delta t^4}{4} & 0 & \\frac{\\Delta t^3}{2} \\\\ \\frac{\\Delta t^3}{2} & 0 & \\Delta t^2 & 0 \\\\ 0 & \\frac{\\Delta t^3}{2} & 0 & \\Delta t^2 \\end{bmatrix}$$

        #### 4. Time Prediction Step
        $$\\hat{\\mathbf{x}}_{t|t-1} = F \\hat{\\mathbf{x}}_{t-1|t-1}$$
        $$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

        #### 5. Joseph-Form Covariance Update
        $$\\mathbf{y}_t = \\mathbf{z}_t - H \\hat{\\mathbf{x}}_{t|t-1}$$
        $$S_t = H P_{t|t-1} H^T + R$$
        $$K_t = P_{t|t-1} H^T S_t^{-1}$$
        $$\\hat{\\mathbf{x}}_{t|t} = \\hat{\\mathbf{x}}_{t|t-1} + K_t \\mathbf{y}_t$$
        $$P_{t|t} = (I - K_t H) P_{t|t-1} (I - K_t H)^T + K_t R K_t^T$$
        """)

    with col_pf:
        st.markdown("""
        ### 🟡 SIR Particle Filter (PF)
        A non-parametric Monte Carlo sequential Bayesian filter approximating arbitrary non-Gaussian posteriors.

        #### 1. Particle Cloud Representation
        $$\\mathcal{S}_t = \\{ X_t^{(i)}, w_t^{(i)} \\}_{i=1}^N, \\quad \\sum_{i=1}^N w_t^{(i)} = 1$$

        #### 2. Vectorized State Propagation
        $$X_t^{(i)} = X_{t-1}^{(i)} F^T + \\mathcal{N}(0, \\Sigma_{proc})$$

        #### 3. Gaussian Likelihood Weighting
        $$p(\\mathbf{z}_t \\mid X_t^{(i)}) = \\frac{1}{2\\pi \\sigma^2} \\exp\\left( -\\frac{\\| [x^{(i)}, y^{(i)}]^T - \\mathbf{z}_t \\|^2}{2\\sigma^2} \\right)$$
        $$w_t^{(i)} \\propto w_{t-1}^{(i)} \\cdot p(\\mathbf{z}_t \\mid X_t^{(i)})$$

        #### 4. Effective Sample Size & Resampling
        $$N_{eff} = \\frac{1}{\\sum_{i=1}^N (w_t^{(i)})^2} < \\frac{N}{2}$$
        - **Systematic Resampling:** Partition $[0, 1)$ with single random offset $u_0 \\sim \\mathcal{U}[0, 1/N)$.
        - **Stratified Resampling:** Draw $u_k \\sim \\mathcal{U}[k/N, (k+1)/N)$.
        - Reset weights: $w_t^{(i)} = 1/N$.

        #### 5. Empirical Weighted Estimates
        $$\\hat{\\mathbf{x}}_t = \\sum_{i=1}^N w_t^{(i)} X_t^{(i)}$$
        $$\\hat{P}_t = \\sum_{i=1}^N w_t^{(i)} (X_t^{(i)} - \\hat{\\mathbf{x}}_t)(X_t^{(i)} - \\hat{\\mathbf{x}}_t)^T$$
        """)

    st.markdown("---")
    st.markdown("""
    ### ⚖️ Architectural & Algorithmic Trade-Offs

    | Dimension | Linear Kalman Filter (KF) | SIR Particle Filter (PF) |
    |---|---|---|
    | **Computational Complexity** | $\\mathcal{O}(d^3)$ per frame ($\sim 0.03\\text{ ms}$) | $\\mathcal{O}(N \\cdot d)$ per frame ($\sim 0.25\\text{ ms}$) |
    | **Linearity Assumption** | Strictly Linear Gaussian ($F, H$) | Fully Non-Linear / Arbitrary Motion |
    | **Multimodal Tracking** | Unimodal only (single Gaussian mode) | Multimodal (tracks simultaneous hypotheses) |
    | **Measurement Dropout** | Pure dead-reckoning continuation | Swarm diffusion across state space |
    | **Implementation** | Closed-form analytical matrix algebra | Vectorized Monte Carlo sampling & resampling |
    | **Practical Throughput** | $\\approx 26,000$ FPS | $\\approx 3,800$ FPS |
    """)


# =========================================================================
# Main Router
# =========================================================================
def main():
    selected_tab = render_sidebar()

    if selected_tab == "🎥 Live Tracking Studio":
        render_live_tracking_studio()
    elif selected_tab == "📊 Benchmarking Lab":
        render_benchmarking_lab()
    elif selected_tab == "📖 Algorithm Formulations":
        render_algorithm_formulations()


if __name__ == "__main__":
    main()
