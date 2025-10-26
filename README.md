

## 🧠 Overview

This repository contains **Assignment 1** of the *Artificial Intelligence Course (Fall 2026)*, featuring two main tasks:

1. 🧩 **10 Types of Machine Learning Optimizers** — Explained in detail in the provided PDF
2. 📈 **Linear Regression using Stochastic Gradient Descent (SGD)** — Implemented **from scratch**

---

## 🎯 Project Description

A complete, educational yet **professionally structured** implementation of **Linear Regression with Stochastic Gradient Descent** — built entirely in **pure Python** (no external ML frameworks).
Includes modern **terminal visualization**, **colorful progress feedback**, and **rich performance charts**.

---

## 📊 Dataset Details

📁 **File:** `MultipleLR-Dataset.csv`
📦 **Samples:** 25  **Columns:** 4  **Features:** 3  **Target:** 1

**Format:**

```
feature1, feature2, feature3, target
73,80,75,152
93,88,93,185
89,91,90,180
96,98,100,196
...
```

---

## ✨ Key Features

### ⚙️ Core Implementation

* 🧮 **Pure Python** (no ML libraries)
* 🔁 **SGD Algorithm** from scratch
* 📏 **Min–Max Normalization** to [0, 1]
* 🔀 **Train/Test Split** (80 / 20)
* 🧠 **Metrics:** R², MSE, MAE

### 🖥️ Professional Terminal Output

* 🎨 **ANSI-colored messages**
* 🔄 **Real-time progress updates**
* ✅ **Emoji status indicators**
* 📜 **Formatted & structured results**

### 📈 Visual Insights

* 📉 **Training Loss Curve**
* 🎯 **Predicted vs Actual (Train/Test)**
* 📊 **Residuals Analysis**
* ⚖️ **Feature Importance**
* 🧩 **Performance Comparison Dashboard**
* 🧠 **7 Elegant Plots** — modern, readable, and polished

---

## 🛠️ Installation

### 🧾 Requirements

* Python 3.7 or newer
* Packages listed in `requirements.txt`

### 💻 Setup

```bash
git clone <repo-url>
cd Assignment-1
pip install -r requirements.txt
```

---

## 🧩 Usage

### 🧠 Professional Mode (Recommended for Submission)

```bash
python main.py
```

> 💎 Full visuals, colorized interface, and detailed analysis

> 🧾 Lightweight output, clear comments, no plots — ideal for discussion

---

## ⚡ Execution Flow

1. 📁 **Data Collection** – Load and validate dataset
2. ✂️ **Preprocessing** – Normalize and split data
3. 🔍 **Exploration** – Quick data overview
4. 🧮 **Model Definition** – Linear Regression + SGD
5. 🚀 **Training** – Gradient updates with live feedback
6. 📊 **Evaluation** – R², MSE, MAE metrics
7. 🔧 **Optimization** – Learning rate tuning
8. 🎯 **Prediction** – Show actual vs predicted results
9. 🎨 **Visualization** – Generate 7 professional charts

---

## 💡 Sample Terminal Output

```
============================================================
📈 Linear Regression with Stochastic Gradient Descent
============================================================

[1] Loading data... ✓ 25 samples loaded

[2] Splitting data (80% train / 20% test)
    ✓ Training: 20 samples
    ✓ Testing : 5 samples

[3] Normalizing features... ✓ Done

[4] Training model...
Iteration 100/1000, Loss = 45.2341
Iteration 200/1000, Loss = 23.5678
...
✓ Model training completed!

============================================================
📊 RESULTS
============================================================

🔧 Model Parameters
  Weights: [0.1234, 0.5678, 0.9012]
  Bias   : 0.3456

📈 Training Set
  R² = 0.8765 | MSE = 12.34 | MAE = 2.79

🎯 Test Set
  R² = 0.8234 | MSE = 15.67 | MAE = 3.12

✨ Visualizing results...
  ✅ Loss Curve
  ✅ Predictions (Train/Test)
  ✅ Residuals
  ✅ Feature Importance
============================================================
🎉 Training completed successfully!
============================================================
```

---


## 🧮 Technical Overview

### ⚙️ Algorithm Details

| Parameter         | Description                   |
| ----------------- | ----------------------------- |
| **Algorithm**     | Stochastic Gradient Descent   |
| **Loss Function** | Mean Squared Error (MSE)      |
| **Learning Rate** | 0.01 (configurable)           |
| **Iterations**    | 1000 (configurable)           |
| **Weights Init.** | Random uniform in [-0.1, 0.1] |
| **Metrics**       | R², MSE, MAE                  |

---

## 🎨 Visualization Dashboard

> 7 fully-styled subplots with professional presentation

|  #  | Visualization              | Description                         |
| :-: | -------------------------- | ----------------------------------- |
| 1️⃣ | **Loss Curve**             | Tracks model convergence            |
| 2️⃣ | **Train Predictions**      | Actual vs Predicted (train data)    |
| 3️⃣ | **Test Predictions**       | Actual vs Predicted (test data)     |
| 4️⃣ | **Residuals**              | Shows prediction errors             |
| 5️⃣ | **Feature Importance**     | Weight magnitudes                   |
| 6️⃣ | **Error Distribution**     | Histogram + mean error line         |
| 7️⃣ | **Performance Comparison** | R² / MSE / MAE across train vs test |

🖌️ **Design Highlights:**

* Consistent **blue–red–orange palette**
* **Emojis + titles** for visual flair
* **Labeled bars/points** with exact values
* **Perfect-prediction lines**
* **Subplot dashboards** with grids for clean readability

---

## 🧠 Advanced Configurations

| Variable               | Default | Description               |
| ---------------------- | ------- | ------------------------- |
| `LEARNING_RATE`        | `0.01`  | Controls update magnitude |
| `N_ITERATIONS`         | `1000`  | Number of training steps  |
| `TEST_SIZE`            | `0.2`   | Train/test ratio          |
| `RANDOM_SEED`          | `42`    | Ensures reproducibility   |
| `SHOW_PLOTS`           | `True`  | Toggle visual output      |
| `PRINT_PROGRESS_EVERY` | `100`   | Log frequency             |

---

🧩 Robustness & Error Handling

* 🧱 **File checks** for dataset existence
* 🔍 **Input validation** for shape & format
* 🛡️ **Safe visualization handling**
* ⏳ **Progress bars & logging**
* 💬 **Descriptive error messages**

---

🧮 Performance Analysis

* 📉 **Convergence tracking**
* 🧠 **Generalization check** on test data
* ⚖️ **Feature weight insights**
* 🔍 **Residual and error analysis**
* 🔧 **Learning rate tuning experiments**
* 🧾 **Statistical validation metrics**

---

✅ Assignment Requirements Met

| Task          | Description                                 | Status                         |
| ------------- | ------------------------------------------- | ------------------------------ |
| 🧩 **Task 1** | *10 ML Optimizers explained*                | ✅ `Assignment-1.pdf`           |
| 📈 **Task 2** | *Linear Regression with SGD (from scratch)* | ✅ `main.py`  |

---

👨‍💻 Author

**👤 Name: Abdelrahman Hamdy Mustafa Ali El-Hamshary
**🎓 Student ID:202002570
**📘 Course:Artificial Intelligence – Fall 2025

---
