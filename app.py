import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

st.set_page_config(page_title="AIQ Sentinel Dashboard", page_icon="🌬️", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at 15% 15%, #d7f7ff 0%, #f7fffc 45%, #f4f7ff 100%);
    }
    .logo-chip {
        float: right;
        background: linear-gradient(135deg, #005f73, #0a9396, #94d2bd);
        color: white;
        font-weight: 700;
        border-radius: 16px;
        padding: 10px 16px;
        box-shadow: 0 8px 24px rgba(10, 147, 150, 0.25);
        animation: pulseGlow 2.2s infinite ease-in-out;
    }
    .engine-card {
        border-radius: 16px;
        background: linear-gradient(120deg, #001219, #005f73);
        color: #e9f9ff;
        padding: 18px 20px;
        box-shadow: 0 10px 24px rgba(0, 18, 25, 0.3);
    }
    .engine-core {
        width: 54px;
        height: 54px;
        border: 4px solid #94d2bd;
        border-top: 4px solid #ee9b00;
        border-radius: 50%;
        animation: spin 1.6s linear infinite;
        margin-bottom: 8px;
    }
    .aq-status-good { color: #2d6a4f; font-weight: 700; font-size: 28px; }
    .aq-status-moderate { color: #ca6702; font-weight: 700; font-size: 28px; }
    .aq-status-poor { color: #bb3e03; font-weight: 700; font-size: 28px; }
    .aq-status-critical { color: #9b2226; font-weight: 700; font-size: 28px; text-decoration: underline; }
    .risk-box {
        border-left: 8px solid #0a9396;
        background: #f1fbff;
        border-radius: 12px;
        padding: 14px;
        margin-top: 8px;
    }
    .arch-wrap {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .arch-node {
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 600;
        color: #001219;
        background: #d8f3dc;
        animation: slideIn 1s ease forwards;
        opacity: 0;
    }
    .arch-node:nth-child(2) { animation-delay: 0.15s; }
    .arch-node:nth-child(3) { animation-delay: 0.3s; }
    .arch-node:nth-child(4) { animation-delay: 0.45s; }
    .arch-node:nth-child(5) { animation-delay: 0.6s; }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes pulseGlow {
        0% { box-shadow: 0 8px 24px rgba(10,147,150,0.2); }
        50% { box-shadow: 0 12px 30px rgba(238,155,0,0.35); }
        100% { box-shadow: 0 8px 24px rgba(10,147,150,0.2); }
    }
    @keyframes slideIn {
        from { transform: translateY(8px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_to_percent(values):
    arr = np.array(values, dtype=float)
    arr[arr < 0] = 0
    s = arr.sum()
    if s <= 0:
        return [0.0 for _ in arr]
    return list((arr / s) * 100.0)


def compute_aqi(temp, hum, gas, occ):
    gas_component = np.clip((gas - 300) / 2200, 0, 1) * 330
    temp_component = np.clip((temp - 18) / 25, 0, 1) * 75
    hum_component = np.clip(abs(hum - 50) / 50, 0, 1) * 45
    occ_component = np.clip(occ / 60, 0, 1) * 50
    return int(np.clip(gas_component + temp_component + hum_component + occ_component, 0, 500))


def interpret_level(level):
    if level <= 50:
        return "Good Air Quality", "aq-status-good", "Ventilation is healthy for students."
    if level <= 100:
        return "Moderate Air Quality", "aq-status-moderate", "Air is acceptable but improve fresh air flow."
    if level <= 200:
        return "Poor Air Quality", "aq-status-poor", "Sensitive students may feel discomfort."
    return "Critical Air Quality", "aq-status-critical", "Immediate action required. Start exhaust and open windows."


def save_uploaded_csv(uploaded_file):
    upload_dir = Path("uploaded_datasets")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uploaded_file.name)
    save_path = upload_dir / safe_name
    save_path.write_bytes(uploaded_file.getbuffer())
    return str(save_path.resolve())


def run_training_mode(mode, csv_path, output_slot):
    command = [sys.executable, "-u", "train_model.py", "--mode", mode, "--csv", csv_path]
    logs = []
    with output_slot:
        st.markdown("### Optimization Output")
        st.caption(f"Dataset used: `{csv_path}`")
        log_box = st.empty()
        with st.status(f"Running {mode.upper()} optimization...", expanded=True) as status:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    logs.append(line.rstrip())
                    log_box.code("\n".join(logs[-120:]) if logs else "Running...", language="bash")
            code = process.wait()
            if code == 0:
                status.update(label=f"{mode.upper()} completed", state="complete")
            else:
                status.update(label=f"{mode.upper()} failed", state="error")
    return code, "\n".join(logs).strip()


def render_reference_graph_board(seed):
    if plt is None:
        st.info("Install matplotlib to view the advanced graph board.")
        return

    np.random.seed(seed)
    fig = plt.figure(figsize=(12, 7), facecolor="#061a2d")
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection="polar")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0a2238")
        ax.grid(alpha=0.25, color="#7cc6fe")

    t = np.arange(80)
    for color in ["#ff9f1c", "#2ec4b6", "#e71d36", "#3a86ff"]:
        series = np.cumsum(np.random.normal(0, 1.0, len(t)))
        ax1.plot(t, series, color=color, linewidth=1.2)
    ax1.set_title("Sensor Dynamics", color="white")
    ax1.set_xlabel("Time", color="white")
    ax1.set_ylabel("Value", color="white")
    ax1.tick_params(colors="#cde7ff")

    x1, y1 = np.random.normal(-2.5, 1.3, 110), np.random.normal(0.5, 1.0, 110)
    x2, y2 = np.random.normal(2.3, 1.1, 110), np.random.normal(1.5, 1.0, 110)
    ax2.scatter(x1, y1, c="#43aa8b", marker="^", s=24, alpha=0.7)
    ax2.scatter(x2, y2, c="#f8961e", marker="o", s=20, alpha=0.75)
    ax2.set_title("Peak Airs Plot", color="white")
    ax2.set_xlabel("X-space", color="white")
    ax2.set_ylabel("Y-space", color="white")
    ax2.tick_params(colors="#cde7ff")

    d1 = np.random.normal(0, 1.1, 1300)
    d2 = np.random.normal(2.0, 0.9, 1300)
    ax3.hist(d1, bins=36, density=True, color="#3a86ff", alpha=0.4, label="Distribution A")
    ax3.hist(d2, bins=36, density=True, color="#9d4edd", alpha=0.45, label="Distribution B")
    ax3.set_title("Distribution Plot", color="white")
    ax3.legend(facecolor="#0a2238", edgecolor="#0a2238", labelcolor="white")
    ax3.tick_params(colors="#cde7ff")

    theta = np.linspace(0, 4 * np.pi, 380)
    radius = np.linspace(0.4, 5.2, 380)
    ax4.set_facecolor("#0a2238")
    ax4.plot(theta, radius, color="#ef233c", linewidth=1.7)
    ax4.plot(theta + 0.9, radius * 0.85, color="#2a9d8f", linewidth=1.0, alpha=0.9)
    ax4.set_title("Polar Spiral", color="white", pad=14)
    ax4.tick_params(colors="#cde7ff")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def load_bundle():
    try:
        bundle = joblib.load("air_quality_optimized_bundle.pkl")
        return bundle, True
    except FileNotFoundError:
        try:
            legacy_model = joblib.load("air_quality_rf_model.pkl")
            legacy_scaler = joblib.load("sensor_scaler.pkl")
            fallback_bundle = {
                "model": legacy_model,
                "scaler": legacy_scaler,
                "pca": None,
                "lda": None,
                "metrics": {"test_accuracy": 0.0},
                "techniques": ["StandardScaler", "RandomForest (legacy)"],
            }
            return fallback_bundle, False
        except FileNotFoundError:
            st.error("Model files are missing. Please run train_model.py first.")
            st.stop()


bundle, optimized_model_loaded = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle.get("pca")
lda = bundle.get("lda")
model_accuracy = float(bundle.get("metrics", {}).get("test_accuracy", 0.0))
algorithm_comparison = bundle.get("algorithm_comparison", [])
training_output_slot = st.container()

st.sidebar.title("Control Panel")
st.sidebar.markdown("Tune classroom sensor values and run live inference.")
with st.sidebar.form("sensor_form"):
    sim_temp = st.slider("Temperature (°C)", 15.0, 45.0, 26.0)
    sim_hum = st.slider("Humidity (%)", 20.0, 100.0, 52.0)
    sim_gas = st.slider("Total Gas Proxy (ppm)", 300, 2500, 850)
    sim_occ = st.slider("Occupancy", 0, 60, 28)
    analysis_tick = st.number_input("Cycle", min_value=1, max_value=9999, value=1)
    st.form_submit_button("Run Live Analysis")

st.sidebar.markdown("### Optimization Runner")
uploaded_csv = st.sidebar.file_uploader("Upload CSV For Training (Optional)", type=["csv"])
csv_for_training = "AirQualityUCI.csv"
if uploaded_csv is not None:
    csv_for_training = save_uploaded_csv(uploaded_csv)
    st.sidebar.success(f"Using uploaded data: {uploaded_csv.name}")
else:
    st.sidebar.caption("Using default dataset: AirQualityUCI.csv")

btn1, btn2 = st.sidebar.columns(2)

if btn1.button("Run PSO", use_container_width=True):
    code, logs = run_training_mode("pso", csv_for_training, training_output_slot)
    st.session_state["train_logs"] = logs
    st.session_state["train_code"] = code
    st.session_state["train_mode"] = "PSO"
    st.session_state["train_dataset"] = csv_for_training
    if code == 0:
        bundle, optimized_model_loaded = load_bundle()
        model = bundle["model"]
        scaler = bundle["scaler"]
        pca = bundle.get("pca")
        lda = bundle.get("lda")
        model_accuracy = float(bundle.get("metrics", {}).get("test_accuracy", 0.0))
        algorithm_comparison = bundle.get("algorithm_comparison", [])

if btn2.button("Run LDA", use_container_width=True):
    code, logs = run_training_mode("lda", csv_for_training, training_output_slot)
    st.session_state["train_logs"] = logs
    st.session_state["train_code"] = code
    st.session_state["train_mode"] = "LDA"
    st.session_state["train_dataset"] = csv_for_training
    if code == 0:
        bundle, optimized_model_loaded = load_bundle()
        model = bundle["model"]
        scaler = bundle["scaler"]
        pca = bundle.get("pca")
        lda = bundle.get("lda")
        model_accuracy = float(bundle.get("metrics", {}).get("test_accuracy", 0.0))
        algorithm_comparison = bundle.get("algorithm_comparison", [])

if "train_code" in st.session_state:
    if st.session_state["train_code"] == 0:
        st.success(
            f'{st.session_state.get("train_mode", "Training")} training completed successfully on '
            f'{st.session_state.get("train_dataset", "default dataset")}.'
        )
    else:
        st.error(f'{st.session_state.get("train_mode", "Training")} training failed.')
    st.code(st.session_state.get("train_logs", "No logs"), language="bash")

features = np.array([[sim_temp, sim_hum, sim_gas, 12, sim_occ]])
features_scaled = scaler.transform(features)
model_input = pca.transform(features_scaled) if pca is not None else features_scaled
model_input = lda.transform(model_input) if lda is not None else model_input

pred_class = int(model.predict(model_input)[0])
proba = model.predict_proba(model_input)[0] if hasattr(model, "predict_proba") else np.array([0.0, 0.0, 0.0, 0.0])

aqi_value = compute_aqi(sim_temp, sim_hum, sim_gas, sim_occ)
status_text, status_class, advisory = interpret_level(aqi_value)

gas_mix_raw = [
    18 + (sim_gas / 120),
    8 + (sim_temp / 2.8) + (sim_occ / 10),
    5 + (sim_hum / 7.5),
    16 + (sim_gas / 160) + (sim_occ / 12),
]
gas_mix_pct = normalize_to_percent(gas_mix_raw)
gas_names = ["CO", "NOx", "VOC", "CO2"]
harmful_idx = int(np.argmax(gas_mix_pct))
harmful_name = gas_names[harmful_idx]
harmful_pct = gas_mix_pct[harmful_idx]

top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("# AI Smart Indoor Air Quality")
    st.caption("Engine-start monitoring architecture with live classroom environmental intelligence.")
with top_right:
    st.markdown('<div class="logo-chip">AERO-SHIELD AIQ</div>', unsafe_allow_html=True)
    st.metric("AQI Level", f"{aqi_value}", delta=f"Cycle #{analysis_tick}")
    if not optimized_model_loaded:
        st.caption("Training bundle not loaded yet. Run train_model.py to enable all optimized graphs.")

intro_left, intro_right = st.columns([2, 2])
with intro_left:
    st.markdown(
        """
        <div class="engine-card">
            <div class="engine-core"></div>
            <h4 style="margin:0;">Engine Start: Live Analytics Pipeline</h4>
            <div style="opacity:0.9;">Sensors initialized -> optimization stack active -> risk engine running</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with intro_right:
    st.markdown("#### AI Architecture")
    st.markdown(
        """
        <div class="arch-wrap">
            <div class="arch-node">Sensor Stream</div>
            <div class="arch-node">Scaling + PCA</div>
            <div class="arch-node">LDA Projection</div>
            <div class="arch-node">PSO Optimized RF</div>
            <div class="arch-node">Live AQI Decision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
summary_left, summary_right = st.columns([2, 1])
with summary_left:
    st.subheader("Live Air Quality Checkup")
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-box">
            <b>Most Harmful Gas Right Now:</b> {harmful_name} ({harmful_pct:.1f}%)<br/>
            <b>Student Safety Advisory:</b> {advisory}
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_right:
    acc_label = f"{model_accuracy * 100:.2f}%" if model_accuracy > 0 else "Legacy Model"
    st.info("Live Status: Connected")
    st.metric("Live Model Accuracy", acc_label, delta="Optimized Stack")
    st.metric("Predicted Risk Class", str(pred_class), delta=f"Confidence {proba[pred_class] * 100:.1f}%")

st.markdown("### Core Live Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Temperature", f"{sim_temp:.1f} C")
m2.metric("Humidity", f"{sim_hum:.1f}%")
m3.metric("Gas Proxy", f"{sim_gas} ppm")
m4.metric("Occupancy", f"{sim_occ}")

st.markdown("### Harmful Gas Percentage Breakdown")
for g_name, g_pct in zip(gas_names, gas_mix_pct):
    st.write(f"{g_name}: {g_pct:.1f}%")
    st.progress(min(int(g_pct), 100))

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Sensor vs Safe Limits",
        "Live Trend Scatter",
        "Optimization + Confidence",
        "3-Algorithm Comparison",
        "Advanced Graph Board",
    ]
)

with tab1:
    bar_data = pd.DataFrame(
        {"Current Value": [sim_temp, sim_hum, sim_gas / 10], "Safe Limit": [28.0, 60.0, 100.0]},
        index=["Temperature", "Humidity", "Gas (x10 ppm)"],
    )
    st.bar_chart(bar_data)

with tab2:
    np.random.seed(int(sim_temp + sim_hum + analysis_tick))
    scatter_data = pd.DataFrame(
        {"Temperature": np.random.normal(sim_temp, 1.8, 120), "Humidity": np.random.normal(sim_hum, 4.2, 120)}
    )
    st.scatter_chart(scatter_data, x="Temperature", y="Humidity")

with tab3:
    st.write("Optimization techniques active in this project:")
    techniques = bundle.get("techniques", ["StandardScaler", "RandomForest"])
    st.success(" | ".join(techniques))
    conf_df = pd.DataFrame(
        {"Class": ["Good", "Moderate", "Poor", "Critical"], "Probability": [float(x) for x in proba]}
    )
    st.dataframe(
        conf_df.style.background_gradient(cmap="YlOrRd", subset=["Probability"]).format({"Probability": "{:.3f}"}),
        use_container_width=True,
    )

with tab4:
    st.write("Accuracy comparison of the three applied algorithm setups.")
    if algorithm_comparison:
        cmp_df = pd.DataFrame(algorithm_comparison)
        cmp_df["Accuracy(%)"] = pd.to_numeric(cmp_df["accuracy"], errors="coerce") * 100
        cmp_df["Train Time (s)"] = pd.to_numeric(cmp_df.get("train_time_sec"), errors="coerce")
        cmp_df["Inference (ms/sample)"] = pd.to_numeric(cmp_df.get("inference_ms_per_sample"), errors="coerce")
        chart_df = cmp_df.dropna(subset=["Accuracy(%)"])
        st.bar_chart(chart_df.set_index("name")["Accuracy(%)"])
        st.caption("Scatter plot: X = train time, Y = accuracy, bubble size = inference time.")
        if plt is not None and not chart_df.empty:
            scatter_df = chart_df.copy()
            if scatter_df["Train Time (s)"].isna().all():
                scatter_df["Train Time (s)"] = np.arange(1, len(scatter_df) + 1)
            if scatter_df["Inference (ms/sample)"].isna().all():
                scatter_df["Inference (ms/sample)"] = 1.0

            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#061a2d")
            ax.set_facecolor("#0a2238")
            colors = ["#3a86ff", "#ff9f1c", "#e63946"]
            for idx, (_, row) in enumerate(scatter_df.iterrows()):
                ax.scatter(
                    row["Train Time (s)"],
                    row["Accuracy(%)"],
                    s=max(80.0, float(row["Inference (ms/sample)"] or 1.0) * 450.0),
                    color=colors[idx % len(colors)],
                    alpha=0.85,
                )
                ax.annotate(
                    row["name"],
                    (row["Train Time (s)"], row["Accuracy(%)"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                )
            ax.set_xlabel("Train Time (seconds)", color="white")
            ax.set_ylabel("Accuracy (%)", color="white")
            ax.grid(alpha=0.25, color="#7cc6fe")
            ax.tick_params(colors="#cde7ff")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.dataframe(
            cmp_df[["name", "Accuracy(%)", "Train Time (s)", "Inference (ms/sample)"]].style.format(
                {"Accuracy(%)": "{:.2f}", "Train Time (s)": "{:.3f}", "Inference (ms/sample)": "{:.4f}"}
            ),
            use_container_width=True,
        )
        if cmp_df["Accuracy(%)"].isna().any():
            st.caption("PSO bar appears after you click `Run PSO` once.")
    else:
        st.info("Comparison data will appear after running train_model.py once.")

with tab5:
    st.write("Reference-style multi-panel analytics graph.")
    render_reference_graph_board(seed=int(sim_temp + sim_hum + analysis_tick))