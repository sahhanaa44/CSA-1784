"""
train_and_export.py
═══════════════════════════════════════════════════════════════════════════════
  Intelligent Elder Fall Detection — TinyML Model Training & Export
  for ESP32 / MPU6050

  AI ALGORITHMS IMPLEMENTED IN THIS FILE
  ───────────────────────────────────────
  [AI-1]  Z-Score Anomaly Detection        (Welford online algorithm)
  [AI-2]  LSTM Neural Network              (2-layer stacked, INT8 quantised)
  [AI-3]  Exponential Moving Average       (post-filter, matches ESP32 logic)
  [AI-4]  Adaptive Threshold Classifier    (noise-adaptive decision boundary)
  [AI-5]  SMOTE — Synthetic Minority Over-sampling Technique  (class balancing)
  [AI-6]  Min-Max Feature Normalisation    (per-feature scaling before LSTM)

  OUTPUTS
  ───────
  lstm_model.tflite   — quantised flatbuffer (~40-80 KB)
  lstm_model.h        — C header for direct inclusion in the Arduino sketch

  USAGE
  ─────
  pip install tensorflow pandas numpy scikit-learn imbalanced-learn matplotlib
  python train_and_export.py --csv your_dataset.csv [--epochs 60] [--plot]

  CSV FORMAT REQUIRED
  ───────────────────
  Columns: ax, ay, az, gx, gy, gz, label
  label: 1 = fall event,  0 = normal activity (ADL)

  FREE DATASETS
  ─────────────
  FallAllD  → https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
  SisFall   → http://sistemic.udea.edu.co/en/research/projects/sisfall/
  MobiAct   → https://bmi.hmu.gr/the-mobiact-dataset-v2-0/
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
SEQ_LEN       = 50       # timesteps per window  (1 s @ 50 Hz)
STEP          = 10       # sliding-window stride
NUM_FEATURES  = 6        # ax ay az gx gy gz
LSTM_UNITS_1  = 32       # first LSTM layer size  (kept small for TinyML)
LSTM_UNITS_2  = 16       # second LSTM layer size
EPOCHS        = 60
BATCH_SIZE    = 64
FALL_LABEL    = 1
EMA_ALPHA     = 0.35     # must match ESP32 sketch value


# ═══════════════════════════════════════════════════════════════════════════
#  [AI-6]  MIN-MAX FEATURE NORMALISATION
#
#  Algorithm:
#      x_norm = (x - x_min) / (x_max - x_min)
#
#  Applied per feature (column) so that all 6 sensor channels occupy
#  the range [0, 1] regardless of unit differences between g and °/s.
#  This prevents the LSTM from being dominated by the larger-magnitude
#  gyroscope readings.
#
#  The fitted min/max values are saved and printed so they can be
#  hard-coded into the ESP32 sketch if online normalisation is needed.
# ═══════════════════════════════════════════════════════════════════════════
class MinMaxNormaliser:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, X_flat: np.ndarray):
        """X_flat: shape (N, NUM_FEATURES) — raw sensor values."""
        self.min_vals = X_flat.min(axis=0)
        self.max_vals = X_flat.max(axis=0)
        print("\n[AI-6 Min-Max Normaliser] Fitted ranges:")
        feat_names = ["ax", "ay", "az", "gx", "gy", "gz"]
        for i, name in enumerate(feat_names):
            print(f"   {name}: [{self.min_vals[i]:.4f}, {self.max_vals[i]:.4f}]")

    def transform(self, X_flat: np.ndarray) -> np.ndarray:
        denom = self.max_vals - self.min_vals
        denom[denom == 0] = 1e-8      # avoid divide-by-zero for static axes
        return (X_flat - self.min_vals) / denom

    def fit_transform(self, X_flat: np.ndarray) -> np.ndarray:
        self.fit(X_flat)
        return self.transform(X_flat)


# ═══════════════════════════════════════════════════════════════════════════
#  [AI-1]  Z-SCORE ANOMALY DETECTION  (Welford online algorithm)
#
#  Algorithm:
#      Maintain a running mean (μ) and variance (σ²) of the accelerometer
#      magnitude using Welford's numerically-stable online update:
#
#          n    ← n + 1
#          δ    ← x − μ
#          μ    ← μ + δ / n
#          M2   ← M2 + δ × (x − μ)        (sum of squared deviations)
#          σ²   ← M2 / (n − 1)
#
#      A sample is labelled "anomalous" when:
#          |x − μ| / σ  >  Z_THRESHOLD
#
#  Used here to: (a) filter the training CSV so the LSTM only learns from
#  windows that actually contain motion anomalies, and (b) mirror the exact
#  pre-filter logic running on the ESP32 (see elder_fall_detection.ino).
# ═══════════════════════════════════════════════════════════════════════════
Z_THRESHOLD      = 2.8
Z_WARMUP_SAMPLES = 100

def zscore_filter_windows(windows: np.ndarray, labels: np.ndarray):
    """
    For each window, compute the peak acceleration magnitude and flag it
    as anomalous using Welford Z-score.  Returns a boolean mask.

    windows: shape (N, SEQ_LEN, NUM_FEATURES)
    """
    n, mean, M2 = 0, 0.0, 0.0
    mask = np.zeros(len(windows), dtype=bool)

    for i, w in enumerate(windows):
        # Peak vector magnitude in this window
        mags   = np.sqrt((w[:, :3] ** 2).sum(axis=1))
        peak   = float(mags.max())

        # Welford update
        n     += 1
        delta  = peak - mean
        mean  += delta / n
        M2    += delta * (peak - mean)

        if n < Z_WARMUP_SAMPLES:
            mask[i] = True           # warm-up: pass all through
            continue

        variance = M2 / (n - 1) if n > 1 else 0.0
        stddev   = float(np.sqrt(variance)) if variance > 0 else 1e-8
        z_score  = abs(peak - mean) / stddev
        mask[i]  = (z_score > Z_THRESHOLD) or (labels[i] == FALL_LABEL)
        # Always keep labelled fall windows regardless of z-score

    n_kept = mask.sum()
    print(f"\n[AI-1 Z-Score Anomaly Filter]  Kept {n_kept}/{len(windows)} windows "
          f"(threshold Z={Z_THRESHOLD})")
    return mask


# ═══════════════════════════════════════════════════════════════════════════
#  [AI-5]  SMOTE — Synthetic Minority Over-sampling Technique
#
#  Algorithm:
#      For each minority-class sample x_i, select k nearest neighbours
#      among the same class.  A synthetic sample is created by:
#
#          x_synthetic = x_i + λ × (x_neighbour − x_i)
#
#      where λ ~ Uniform(0, 1).  This interpolates between real samples
#      in feature space rather than simply duplicating them, resulting in
#      a richer and less over-fitted augmented dataset.
#
#  Why: fall events are rare — datasets are typically 5–15% fall, 85–95%
#  normal ADL.  A classifier trained on this imbalance learns to always
#  predict "normal" for high accuracy but zero recall on falls.
#  SMOTE balances the classes before training.
#
#  Note: SMOTE is applied in the flattened feature space
#  (N, SEQ_LEN × NUM_FEATURES) then reshaped back to (N, SEQ_LEN, NUM_FEATURES).
# ═══════════════════════════════════════════════════════════════════════════
def smote_balance(X: np.ndarray, y: np.ndarray, k_neighbours: int = 5,
                  random_state: int = 42) -> tuple:
    """
    Custom lightweight SMOTE for 3-D sequence data.
    X: (N, SEQ_LEN, NUM_FEATURES)
    y: (N,)
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_count  = counts.max()

    X_aug = [X.copy()]
    y_aug = [y.copy()]

    for cls, count in zip(classes, counts):
        if count >= majority_count:
            continue   # skip the majority class

        minority_idx = np.where(y == cls)[0]
        X_min = X[minority_idx]                           # shape (M, SEQ, FEAT)
        X_flat = X_min.reshape(len(X_min), -1)            # (M, SEQ*FEAT)
        needed = majority_count - count

        synthetic = []
        for _ in range(needed):
            # Pick a random minority sample
            i = rng.integers(0, len(X_flat))
            xi = X_flat[i]

            # Find k nearest neighbours (Euclidean) in flattened space
            dists = np.linalg.norm(X_flat - xi, axis=1)
            dists[i] = np.inf                              # exclude self
            nn_idx = np.argpartition(dists, k_neighbours)[:k_neighbours]
            j = rng.choice(nn_idx)
            xj = X_flat[j]

            # Interpolate
            lam = rng.uniform(0, 1)
            x_syn = xi + lam * (xj - xi)
            synthetic.append(x_syn.reshape(SEQ_LEN, NUM_FEATURES))

        X_aug.append(np.array(synthetic, dtype=np.float32))
        y_aug.append(np.full(len(synthetic), cls, dtype=np.int32))
        print(f"[AI-5 SMOTE] Class {cls}: generated {needed} synthetic samples "
              f"(k={k_neighbours})")

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)

    # Shuffle
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & WINDOWING
# ═══════════════════════════════════════════════════════════════════════════
def load_and_window(csv_path: str, normaliser: MinMaxNormaliser):
    df = pd.read_csv(csv_path)
    required = {"ax", "ay", "az", "gx", "gy", "gz", "label"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"[Error] CSV missing columns: {missing}")

    features_raw = df[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)
    labels_raw   = df["label"].values.astype(np.int32)

    # [AI-6] Apply min-max normalisation to the raw signal
    features_norm = normaliser.fit_transform(features_raw)

    # Sliding window segmentation
    X, y = [], []
    for i in range(0, len(features_norm) - SEQ_LEN, STEP):
        window = features_norm[i: i + SEQ_LEN]
        window_labels = labels_raw[i: i + SEQ_LEN]
        X.append(window)
        y.append(1 if window_labels.mean() >= 0.5 else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n[Data] Total windows: {len(X)}  "
          f"(fall={y.sum()}, non-fall={(y==0).sum()})")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
#  [AI-2]  LSTM NEURAL NETWORK ARCHITECTURE
#
#  Architecture:
#      Input      →  [batch=1, timesteps=50, features=6]
#      LSTM-1     →  32 units, return_sequences=True   (captures short patterns)
#      Dropout    →  30% (prevents over-fitting)
#      LSTM-2     →  16 units, return_sequences=False  (captures long patterns)
#      Dropout    →  30%
#      Dense-1    →  16 units, ReLU activation
#      Dense-out  →  2 units,  Softmax  → [P(non-fall), P(fall)]
#
#  Why LSTM:
#      Falls are temporal events — a fall involves a free-fall phase
#      (~150 ms), an impact spike, and a post-impact stillness.
#      LSTMs maintain hidden state across the 50-sample sequence so they
#      learn this temporal pattern rather than treating each timestep
#      independently (as a simple Dense network would).
#
#  Quantisation (INT8):
#      The trained float32 model is converted to INT8 using TFLite's
#      post-training full-integer quantisation.  This:
#        • Reduces model size by ~4×
#        • Reduces inference time on ESP32 by ~2–3×
#        • Fits within the ~320 KB SRAM of the ESP32
# ═══════════════════════════════════════════════════════════════════════════
def build_lstm_model() -> tf.keras.Model:
    model = models.Sequential([
        # ── [AI-2] LSTM Layer 1: learns short-term motion patterns ──────────
        layers.LSTM(LSTM_UNITS_1,
                    input_shape=(SEQ_LEN, NUM_FEATURES),
                    return_sequences=True,
                    name="LSTM_Layer1_shortterm"),
        layers.BatchNormalization(name="BN_1"),
        layers.Dropout(0.30, name="Dropout_1"),

        # ── [AI-2] LSTM Layer 2: learns long-term temporal context ──────────
        layers.LSTM(LSTM_UNITS_2,
                    return_sequences=False,
                    name="LSTM_Layer2_longterm"),
        layers.BatchNormalization(name="BN_2"),
        layers.Dropout(0.30, name="Dropout_2"),

        # ── Dense classifier ─────────────────────────────────────────────────
        layers.Dense(16, activation="relu", name="Dense_Hidden"),
        layers.Dense(2,  activation="softmax", name="Output_Softmax"),
        #                             └── outputs [P(non-fall), P(fall)]
    ], name="LSTM_FallDetector")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  [AI-3]  EMA POST-PROCESSING — VALIDATION
#
#  After LSTM inference on the test set, apply the same EMA smoothing
#  that runs on the ESP32 to measure its real-world impact on metrics.
#
#  Formula:  ema(t) = α × p(t)  +  (1 − α) × ema(t-1)
# ═══════════════════════════════════════════════════════════════════════════
def apply_ema_to_probabilities(probs: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """
    probs: 1-D array of raw LSTM fall probabilities [P(fall)] per window.
    Returns EMA-smoothed probabilities.
    """
    smoothed = np.zeros_like(probs)
    ema = 0.0
    for i, p in enumerate(probs):
        ema = alpha * p + (1.0 - alpha) * ema
        smoothed[i] = ema
    return smoothed


# ═══════════════════════════════════════════════════════════════════════════
#  [AI-4]  ADAPTIVE THRESHOLD CLASSIFIER — VALIDATION
#
#  Formula:  T = BASE + NOISE_GAIN × ambient_noise
#
#  The ambient noise estimate is updated with a slow exponential decay
#  on non-fall windows (mirrors the ESP32 implementation exactly).
# ═══════════════════════════════════════════════════════════════════════════
BASE_THRESHOLD = 0.72
NOISE_GAIN     = 0.20
NOISE_DECAY    = 0.995

def apply_adaptive_threshold(smoothed_probs: np.ndarray, labels: np.ndarray):
    """
    Simulate the adaptive threshold classifier on the test set.
    Returns predicted labels and per-sample thresholds.
    """
    ambient = 0.0
    preds      = np.zeros(len(smoothed_probs), dtype=np.int32)
    thresholds = np.zeros(len(smoothed_probs), dtype=np.float32)

    for i, ema_p in enumerate(smoothed_probs):
        T = min(BASE_THRESHOLD + NOISE_GAIN * ambient, 0.93)
        thresholds[i] = T
        preds[i]      = 1 if ema_p >= T else 0

        # Update noise estimate only on non-fall windows
        if ema_p < BASE_THRESHOLD:
            ambient = ambient * NOISE_DECAY + (1.0 - NOISE_DECAY) * ema_p

    return preds, thresholds


# ═══════════════════════════════════════════════════════════════════════════
#  TFLITE EXPORT (INT8 quantised)
# ═══════════════════════════════════════════════════════════════════════════
def export_tflite(keras_model: tf.keras.Model,
                  X_train: np.ndarray,
                  out_path: str = "lstm_model.tflite") -> bytes:
    """
    Full-integer post-training quantisation.
    The representative dataset is a random 500-sample subset of X_train;
    TFLite uses it to calibrate the INT8 activation ranges.
    """
    idx  = np.random.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    calib = X_train[idx]

    def representative_dataset():
        for sample in calib:
            yield [sample[np.newaxis].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations            = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset   = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type     = tf.int8
    conv.inference_output_type    = tf.int8

    tflite_bytes = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"\n[Export] {out_path}  ({len(tflite_bytes)/1024:.1f} KB)")
    return tflite_bytes


def export_c_header(tflite_bytes: bytes,
                    var_name:    str = "lstm_model_tflite",
                    out_path:    str = "lstm_model.h"):
    """Convert TFLite binary to a C byte-array header for the Arduino sketch."""
    hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    content = (
        "// Auto-generated by train_and_export.py — DO NOT EDIT MANUALLY\n"
        "// AI algorithms: Z-Score [AI-1], LSTM [AI-2], EMA [AI-3],\n"
        "//                Adaptive Threshold [AI-4], SMOTE [AI-5], Min-Max Norm [AI-6]\n"
        "//\n"
        "// Copy this file into the same folder as elder_fall_detection.ino\n\n"
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        f"const unsigned char {var_name}[] = {{\n  {hex_vals}\n}};\n\n"
        f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n"
    )
    with open(out_path, "w") as f:
        f.write(content)
    print(f"[Export] {out_path}  ({len(tflite_bytes)} bytes embedded)")


# ═══════════════════════════════════════════════════════════════════════════
#  OPTIONAL PLOTS
# ═══════════════════════════════════════════════════════════════════════════
def save_training_plots(history, smoothed_probs, labels_test, preds, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Elder Fall Detection — AI Training Results", fontsize=14)

    # 1. Training curves
    ax = axes[0]
    ax.plot(history.history["accuracy"],     label="Train Acc")
    ax.plot(history.history["val_accuracy"], label="Val Acc")
    ax.plot(history.history["loss"],         label="Train Loss", linestyle="--")
    ax.plot(history.history["val_loss"],     label="Val Loss",   linestyle="--")
    ax.set_title("[AI-2] LSTM Training Curves")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # 2. EMA-smoothed probabilities on test set
    ax = axes[1]
    colors = ["blue" if l == 0 else "red" for l in labels_test]
    ax.scatter(range(len(smoothed_probs)), smoothed_probs, c=colors, s=8, alpha=0.6)
    ax.plot(thresholds, color="orange", linewidth=1.5, label="Adaptive threshold [AI-4]")
    ax.axhline(BASE_THRESHOLD, color="green", linestyle="--", linewidth=1,
               label=f"Base threshold ({BASE_THRESHOLD})")
    ax.set_title("[AI-3] EMA Probabilities + [AI-4] Adaptive Threshold")
    ax.set_xlabel("Window index"); ax.set_ylabel("P(fall)")
    ax.legend(fontsize=8); ax.grid(True)

    # 3. Confusion matrix
    ax = axes[2]
    cm = confusion_matrix(labels_test, preds)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["non-fall", "fall"])
    ax.set_yticklabels(["non-fall", "fall"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title("Confusion Matrix (with [AI-3]+[AI-4])")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=120)
    print("[Plot] Saved training_results.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train TinyML LSTM fall detector and export to ESP32 header")
    parser.add_argument("--csv",    required=True, help="Path to labeled IMU CSV")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--plot",   action="store_true",
                        help="Save training_results.png after training")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("  Elder Fall Detection — TinyML LSTM Training Pipeline")
    print("  AI algorithms: Z-Score [1] · LSTM [2] · EMA [3]")
    print("                 AdaptiveThresh [4] · SMOTE [5] · MinMax [6]")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Step 1: Load + [AI-6] normalise + window ─────────────────────────────
    norm = MinMaxNormaliser()
    X, y = load_and_window(args.csv, norm)

    # ── Step 2: [AI-1] Z-Score filter — keep only anomalous windows ──────────
    mask = zscore_filter_windows(X, y)
    X, y = X[mask], y[mask]
    print(f"[AI-1] After z-score filtering: {len(X)} windows")

    # ── Step 3: Train/test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\n[Split] Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Step 4: [AI-5] SMOTE balance training set ────────────────────────────
    X_train, y_train = smote_balance(X_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[AI-5] After SMOTE — class distribution: {dict(zip(unique, counts))}")

    # ── Step 5: [AI-2] Build and train LSTM ──────────────────────────────────
    print("\n[AI-2] Building LSTM model...")
    model = build_lstm_model()

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=cb,
    )

    # ── Step 6: Raw LSTM evaluation ──────────────────────────────────────────
    print("\n[AI-2 LSTM] Raw test evaluation:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Accuracy: {acc*100:.2f}%   Loss: {loss:.4f}")

    raw_probs = model.predict(X_test, verbose=0)[:, 1]   # P(fall)

    # ── Step 7: [AI-3] EMA smoothing on test probabilities ───────────────────
    print("\n[AI-3] Applying EMA smoothing (α={:.2f})...".format(EMA_ALPHA))
    smoothed = apply_ema_to_probabilities(raw_probs, alpha=EMA_ALPHA)

    # ── Step 8: [AI-4] Adaptive threshold classification ─────────────────────
    print("[AI-4] Applying adaptive threshold classifier...")
    preds, thresholds = apply_adaptive_threshold(smoothed, y_test)

    print("\n[Full Pipeline] Classification Report (LSTM + EMA + Adaptive Threshold):")
    print(classification_report(y_test, preds,
                                target_names=["non-fall", "fall"]))

    # ── Step 9: Export ───────────────────────────────────────────────────────
    tflite_bytes = export_tflite(model, X_train)
    export_c_header(tflite_bytes)

    # ── Step 10: Optional plots ───────────────────────────────────────────────
    if args.plot:
        save_training_plots(history, smoothed, y_test, preds, thresholds)

    print("\n✅  All done!")
    print("    Copy lstm_model.h into your Arduino sketch folder and upload.")


if __name__ == "__main__":
    main()
