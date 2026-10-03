import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)

# Load dataset
start = time.perf_counter()
df = pd.read_csv("creditcard.csv")
load_seconds = time.perf_counter() - start

X = df.drop(columns=["Class"])
y = df["Class"]

# 60% train, 20% validation, 20% test.
# Stratify giữ tỷ lệ giao dịch gian lận trong từng tập.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25, stratify=y_trainval, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)

# Train, dừng sớm dựa trên validation AUC.
start = time.perf_counter()
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(50, first_metric_only=True),
        lgb.log_evaluation(100),
    ],
)
training_seconds = time.perf_counter() - start

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Warm-up trước khi đo inference.
one_row = X_test.iloc[:1]
batch = X_test.iloc[:1000]
model.predict_proba(one_row)
model.predict_proba(batch)

single_times = []
for _ in range(100):
    start = time.perf_counter()
    model.predict_proba(one_row)
    single_times.append(time.perf_counter() - start)

batch_times = []
for _ in range(20):
    start = time.perf_counter()
    model.predict_proba(batch)
    batch_times.append(time.perf_counter() - start)

batch_seconds = float(np.mean(batch_times))
result = {
    "dataset_rows": len(df),
    "fraud_rows": int(y.sum()),
    "train_rows": len(X_train),
    "validation_rows": len(X_valid),
    "test_rows": len(X_test),
    "load_data_seconds": load_seconds,
    "training_seconds": training_seconds,
    "best_iteration": int(model.best_iteration_),
    "auc_roc": float(roc_auc_score(y_test, probabilities)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "inference_latency_1_row_ms": float(np.mean(single_times) * 1000),
    "inference_batch_rows": len(batch),
    "inference_1000_rows_ms": batch_seconds * 1000,
    "inference_throughput_rows_per_second": len(batch) / batch_seconds,
    "classification_threshold": 0.5,
    "inference_threads": 2,
}

output = json.dumps(result, indent=2)
Path("benchmark_result.json").write_text(output + "\n", encoding="utf-8")
model.booster_.save_model("lightgbm_model.txt")

print("\n=== BENCHMARK RESULT ===")
print(output)
print("\nSaved: benchmark_result.json, lightgbm_model.txt")
