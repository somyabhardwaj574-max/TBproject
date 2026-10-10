import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "tb_model.keras")
IMG_SIZE = (150, 150)

# ── 1. Recreate Exact 15% Unseen Test Set (Pure Python) ───
random.seed(42)
np.random.seed(42)

normal_files = [os.path.join(DATASET_DIR, "Normal", f) for f in os.listdir(os.path.join(DATASET_DIR, "Normal"))]
tb_files = [os.path.join(DATASET_DIR, "TB", f) for f in os.listdir(os.path.join(DATASET_DIR, "TB"))]

random.shuffle(normal_files)
random.shuffle(tb_files)

# Last 15% was strictly held out for test
test_paths = normal_files[int(len(normal_files) * 0.85):] + tb_files[int(len(tb_files) * 0.85):]
y_true = np.array([0] * len(normal_files[int(len(normal_files) * 0.85):]) + [1] * len(tb_files[int(len(tb_files) * 0.85):]))

print(f"Test Set Count: Normal={(y_true == 0).sum()}, TB={(y_true == 1).sum()}")

# ── 2. Load and Predict ──────────────────────────────────
X_test = np.array([image.img_to_array(image.load_img(p, target_size=IMG_SIZE)) / 255.0 for p in test_paths])
model = load_model(MODEL_PATH)
y_prob = model.predict(X_test, batch_size=32).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# ── 3. Pure NumPy Evaluation Metrics ─────────────────────
TP = int(np.sum((y_true == 1) & (y_pred == 1)))
TN = int(np.sum((y_true == 0) & (y_pred == 0)))
FP = int(np.sum((y_true == 0) & (y_pred == 1)))
FN = int(np.sum((y_true == 1) & (y_pred == 0)))

accuracy = (TP + TN) / len(y_true)
precision_tb = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_tb = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_tb = 2 * (precision_tb * recall_tb) / (precision_tb + recall_tb) if (precision_tb + recall_tb) > 0 else 0

precision_norm = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_norm = TN / (TN + FP) if (TN + FP) > 0 else 0
f1_norm = 2 * (precision_norm * recall_norm) / (precision_norm + recall_norm) if (precision_norm + recall_norm) > 0 else 0

print("\n" + "=" * 45)
print("Classification Report (Pure NumPy)")
print("=" * 45)
print(f"Class        Precision  Recall  F1-Score  Support")
print(f"Normal          {precision_norm:.2f}     {recall_norm:.2f}     {f1_norm:.2f}      {(y_true == 0).sum()}")
print(f"TB              {precision_tb:.2f}     {recall_tb:.2f}     {f1_tb:.2f}      {(y_true == 1).sum()}")
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

print("\n" + "=" * 45)
print("Confusion Matrix")
print("=" * 45)
print(f"[[TN={TN}, FP={FP}],")
print(f" [FN={FN}, TP={TP}]]\n")

# ── 4. Pure NumPy ROC Curve Calculation ──────────────────
thresholds = np.linspace(0.0, 1.0, 200)
fpr_list, tpr_list = [], []

P = float((y_true == 1).sum())
N = float((y_true == 0).sum())

for t in thresholds:
    pred = (y_prob >= t).astype(int)
    tpr_list.append(np.sum((y_true == 1) & (pred == 1)) / P)
    fpr_list.append(np.sum((y_true == 0) & (pred == 1)) / N)

fpr_arr = np.array(fpr_list)[::-1]
tpr_arr = np.array(tpr_list)[::-1]

# Trapezoidal rule for AUC
auc_score = -np.trapezoid(tpr_arr, fpr_arr) if hasattr(np, 'trapezoid') else -np.trapz(tpr_arr, fpr_arr)
auc_val = abs(float(auc_score))

# ── 5. Plot Standard ROC Curve ───────────────────────────
plt.figure(figsize=(7, 6))
plt.plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC) - VGG16", fontweight="bold")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "roc_curve_standard.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Standard ROC AUC: {auc_val:.4f}")
print(f"Saved ROC curve image to: {out_path}")