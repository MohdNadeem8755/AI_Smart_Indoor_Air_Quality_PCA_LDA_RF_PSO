import argparse
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def determine_quality(temp, hum, gas):
    if gas < 1000 and temp < 28 and hum < 60:
        return 0
    if gas < 1250 and temp < 32:
        return 1
    if gas < 1500:
        return 2
    return 3


def _norm_col(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(df, candidates, required=True):
    normalized = {_norm_col(col): col for col in df.columns}
    for candidate in candidates:
        c_norm = _norm_col(candidate)
        if c_norm in normalized:
            return normalized[c_norm]
    if required:
        raise ValueError(
            f"Required column not found. Tried {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def load_and_prepare_dataset(csv_path):
    print(f"Step 1: Loading dataset from: {csv_path}", flush=True)
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=";")

    print("Step 2: Cleaning data...", flush=True)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")

    temp_col = _find_column(df, ["T", "Temperature", "Temp", "temp"])
    hum_col = _find_column(df, ["RH", "Humidity", "Hum", "humidity"])
    gas_col = _find_column(df, ["PT08.S1(CO)", "Gas_Concentration", "Gas", "CO2", "co2", "Gas Level", "gas"])
    time_col = _find_column(df, ["Time", "Hour", "Timestamp", "Datetime"], required=False)
    occ_col = _find_column(df, ["Occupancy", "People", "Students", "Count"], required=False)

    prepared = pd.DataFrame()
    prepared["Temperature"] = pd.to_numeric(
        df[temp_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    prepared["Humidity"] = pd.to_numeric(
        df[hum_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    prepared["Gas_Concentration"] = pd.to_numeric(
        df[gas_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )

    if time_col is not None:
        time_part = df[time_col].astype(str).str.slice(0, 2)
        prepared["Time_of_Day"] = pd.to_numeric(time_part, errors="coerce").fillna(12).astype(int)
    else:
        prepared["Time_of_Day"] = 12

    if occ_col is not None:
        occ_values = pd.to_numeric(df[occ_col], errors="coerce")
        prepared["Occupancy"] = occ_values.fillna(occ_values.median() if not occ_values.dropna().empty else 25).astype(int)
    else:
        np.random.seed(42)
        prepared["Occupancy"] = np.random.randint(0, 60, size=len(prepared))

    prepared = prepared.replace(-200.0, np.nan)
    prepared = prepared.dropna(subset=["Temperature", "Humidity", "Gas_Concentration"])

    if len(prepared) < 20:
        raise ValueError("Dataset has too few valid rows after cleaning. Please upload a cleaner CSV.")

    print("Step 3: Feature processing...", flush=True)
    prepared["Air_Quality_Class"] = prepared.apply(
        lambda row: determine_quality(row["Temperature"], row["Humidity"], row["Gas_Concentration"]), axis=1
    )
    return prepared


def pso_optimize_rf(X_train, y_train, n_particles=4, n_iter=4):
    print("Step 4: PSO swarm optimization for RandomForest...", flush=True)
    bounds = np.array(
        [
            [80, 180],  # n_estimators
            [6, 20],  # max_depth
            [2, 12],  # min_samples_split
            [1, 6],  # min_samples_leaf
        ],
        dtype=float,
    )

    np.random.seed(42)
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, bounds.shape[0]))
    velocities = np.zeros_like(particles)

    pbest_positions = particles.copy()
    pbest_scores = np.full(n_particles, -np.inf)
    gbest_position = particles[0].copy()
    gbest_score = -np.inf

    w, c1, c2 = 0.7, 1.4, 1.4

    def evaluate_particle(params):
        n_estimators = int(np.clip(round(params[0]), bounds[0, 0], bounds[0, 1]))
        max_depth = int(np.clip(round(params[1]), bounds[1, 0], bounds[1, 1]))
        min_samples_split = int(np.clip(round(params[2]), bounds[2, 0], bounds[2, 1]))
        min_samples_leaf = int(np.clip(round(params[3]), bounds[3, 0], bounds[3, 1]))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        cv_scores = cross_val_score(clf, X_train, y_train, cv=2, scoring="accuracy")
        return cv_scores.mean()

    for iteration in range(n_iter):
        for i in range(n_particles):
            score = evaluate_particle(particles[i])
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = particles[i].copy()
            if score > gbest_score:
                gbest_score = score
                gbest_position = particles[i].copy()

        for i in range(n_particles):
            r1 = np.random.rand(bounds.shape[0])
            r2 = np.random.rand(bounds.shape[0])
            cognitive = c1 * r1 * (pbest_positions[i] - particles[i])
            social = c2 * r2 * (gbest_position - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social
            particles[i] = np.clip(particles[i] + velocities[i], bounds[:, 0], bounds[:, 1])

        print(f"  PSO iteration {iteration + 1}/{n_iter} best CV accuracy: {gbest_score:.4f}", flush=True)

    best_params = {
        "n_estimators": int(round(gbest_position[0])),
        "max_depth": int(round(gbest_position[1])),
        "min_samples_split": int(round(gbest_position[2])),
        "min_samples_leaf": int(round(gbest_position[3])),
        "random_state": 42,
        "n_jobs": -1,
    }
    return best_params, gbest_score


def train_pipeline(mode="pso", csv_path="AirQualityUCI.csv"):
    df = load_and_prepare_dataset(csv_path)

    X = df[["Temperature", "Humidity", "Gas_Concentration", "Time_of_Day", "Occupancy"]]
    y = df["Air_Quality_Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Step 4: Applying StandardScaler + PCA + LDA...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Step 4A: Training baseline RandomForest (scaled features only)...", flush=True)
    t0 = time.perf_counter()
    baseline_rf = RandomForestClassifier(n_estimators=120, max_depth=18, random_state=42, n_jobs=-1)
    baseline_rf.fit(X_train_scaled, y_train)
    baseline_train_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    baseline_pred = baseline_rf.predict(X_test_scaled)
    baseline_infer_ms = ((time.perf_counter() - t0) / max(1, len(X_test_scaled))) * 1000.0
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # PCA first reduces noise/redundancy.
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # LDA maximizes class separation before classifier training.
    lda = LinearDiscriminantAnalysis(n_components=min(3, y.nunique() - 1))
    X_train_lda = lda.fit_transform(X_train_pca, y_train)
    X_test_lda = lda.transform(X_test_pca)

    print("Step 4B: Training RF on PCA+LDA features (without PSO)...", flush=True)
    t0 = time.perf_counter()
    lda_rf = RandomForestClassifier(n_estimators=120, max_depth=18, random_state=42, n_jobs=-1)
    lda_rf.fit(X_train_lda, y_train)
    lda_train_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    lda_pred = lda_rf.predict(X_test_lda)
    lda_infer_ms = ((time.perf_counter() - t0) / max(1, len(X_test_lda))) * 1000.0
    lda_acc = accuracy_score(y_test, lda_pred)

    selected_model = lda_rf
    selected_algorithm = "RF + PCA + LDA"
    selected_acc = float(lda_acc)
    best_params = None
    best_cv = None
    pso_acc = None
    pso_train_sec = None
    pso_infer_ms = None

    if mode == "pso":
        t0 = time.perf_counter()
        best_params, best_cv = pso_optimize_rf(X_train_lda, y_train)
        pso_tune_sec = time.perf_counter() - t0
        print(f"Best params from PSO: {best_params}", flush=True)

        print("Step 5: Training final PSO-optimized RandomForest...", flush=True)
        t1 = time.perf_counter()
        rf = RandomForestClassifier(**best_params)
        rf.fit(X_train_lda, y_train)
        rf_fit_sec = time.perf_counter() - t1

        t2 = time.perf_counter()
        y_pred = rf.predict(X_test_lda)
        pso_infer_ms = ((time.perf_counter() - t2) / max(1, len(X_test_lda))) * 1000.0
        pso_acc = accuracy_score(y_test, y_pred)
        pso_train_sec = pso_tune_sec + rf_fit_sec
        selected_model = rf
        selected_algorithm = "PSO + PCA + LDA + RF"
        selected_acc = float(pso_acc)
        print(f"Test accuracy: {pso_acc:.4f}", flush=True)
    else:
        print("Step 5: Using LDA pipeline model (PSO skipped by user choice).", flush=True)
        print(f"Test accuracy: {selected_acc:.4f}", flush=True)

    print("Step 6: Saving model artifacts...", flush=True)
    bundle = {
        "model": selected_model,
        "scaler": scaler,
        "pca": pca,
        "lda": lda,
        "selected_algorithm": selected_algorithm,
        "metrics": {
            "cv_accuracy": float(best_cv) if best_cv is not None else None,
            "test_accuracy": float(selected_acc),
            "baseline_accuracy": float(baseline_acc),
            "lda_accuracy": float(lda_acc),
            "pso_accuracy": float(pso_acc) if pso_acc is not None else None,
        },
        "algorithm_comparison": [
            {
                "name": "RF (Scaled Baseline)",
                "accuracy": float(baseline_acc),
                "train_time_sec": float(baseline_train_sec),
                "inference_ms_per_sample": float(baseline_infer_ms),
            },
            {
                "name": "RF + PCA + LDA",
                "accuracy": float(lda_acc),
                "train_time_sec": float(lda_train_sec),
                "inference_ms_per_sample": float(lda_infer_ms),
            },
            {
                "name": "PSO + PCA + LDA + RF",
                "accuracy": float(pso_acc) if pso_acc is not None else None,
                "train_time_sec": float(pso_train_sec) if pso_train_sec is not None else None,
                "inference_ms_per_sample": float(pso_infer_ms) if pso_infer_ms is not None else None,
            },
        ],
        "params": best_params,
        "techniques": ["Swarm Intelligence (PSO)", "PCA", "LDA", "RandomForest"],
    }
    joblib.dump(bundle, "air_quality_optimized_bundle.pkl")

    # Legacy files kept for backward compatibility.
    joblib.dump(selected_model, "air_quality_rf_model.pkl")
    joblib.dump(scaler, "sensor_scaler.pkl")

    print(f"\nSUCCESS: Optimized AI pipeline saved. Active model: {selected_algorithm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pso", "lda"], default="pso")
    parser.add_argument("--csv", default="AirQualityUCI.csv")
    args = parser.parse_args()
    train_pipeline(mode=args.mode, csv_path=args.csv)