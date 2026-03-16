# PCOS Detection using Explainable Deep Learning Ensemble

Early detection of **Polycystic Ovary Syndrome (PCOS)** using a **genetic algorithm optimized deep learning ensemble with explainable AI (SHAP)**.

This project proposes an interpretable machine learning framework that combines **feature optimization, deep neural networks, ensemble learning, and explainability** to improve the accuracy and transparency of PCOS diagnosis using structured clinical data.

---

# Overview

Polycystic Ovary Syndrome (PCOS) is one of the most common endocrine disorders affecting women of reproductive age. Due to **heterogeneous symptoms and varying diagnostic criteria**, PCOS often remains underdiagnosed.

This project addresses the problem by building a **data-driven diagnostic system** that:

* Uses **genetic algorithms for feature selection**
* Combines **multiple deep neural networks**
* Applies **LogitBoost ensemble learning**
* Provides **clinical interpretability using SHAP**

The proposed model achieved an **accuracy of 91.46%**, outperforming several traditional machine learning models and stacking ensembles.

---

# Key Features

* Genetic Algorithm based feature selection
* Deep Learning MLP Ensemble
* LogitBoost meta-learner
* SMOTE based class imbalance handling
* SHAP explainable AI for model interpretability
* Structured clinical dataset based prediction
* Modular and extensible ML pipeline

---

# Methodology

The system follows the pipeline below:

```
Dataset
   ↓
Data Preprocessing
   ↓
SMOTE Class Balancing
   ↓
Genetic Algorithm Feature Selection
   ↓
MLP Ensemble Training
   ↓
LogitBoost Meta Learning
   ↓
Prediction
   ↓
SHAP Explainability
```

---

# Model Architecture

The ensemble consists of two deep neural networks:

### MLP-1

* Hidden layers: 2
* Neurons: 64 each
* Activation: ReLU
* Dropout: 0.3
* Optimizer: Adam

### MLP-2

* Hidden layers: 3
* Neurons: 96 each
* Activation: ReLU
* Dropout: 0.3
* Optimizer: Adam

The outputs of these models are combined using **LogitBoost**, which assigns weights based on prediction errors.

---

# Dataset

The dataset contains **structured clinical and biochemical attributes** related to PCOS.

Example features include:

* Follicle Count
* Cycle Regularity
* Hair Growth
* Weight Gain
* Hormonal Indicators
* Lifestyle attributes

The dataset was preprocessed using:

* Missing value imputation
* Label encoding
* Z-score normalization
* Correlation analysis

---

# Feature Selection

A **Genetic Algorithm (GA)** was used to select the optimal subset of features.

The fitness function balances:

* Model accuracy
* Number of selected features
* Feature diversity

```
Fitness = α * Accuracy − β * Feature_Count + γ * Diversity
```

This ensures the model remains **accurate while avoiding unnecessary features**.

---

# Explainable AI

The model integrates **SHAP (SHapley Additive Explanations)** to improve interpretability.

SHAP provides:

* Global feature importance
* Instance-level prediction explanations
* Clinical interpretability

Key influential features identified include:

* Follicle Count
* Hair Growth
* Cycle Irregularity

---

# Results

| Model            | Accuracy   | Precision  | Recall     | F1 Score   |
| ---------------- | ---------- | ---------- | ---------- | ---------- |
| XGBoost          | 0.9024     | 0.9017     | 0.9024     | 0.9014     |
| Stack-1          | 0.9024     | 0.9100     | 0.9024     | 0.9040     |
| Stack-2          | 0.8780     | 0.8767     | 0.8780     | 0.8768     |
| **MLP Ensemble** | **0.9146** | **0.9141** | **0.9146** | **0.9142** |

The proposed ensemble achieved the **best overall performance**.

---

# Technology Stack

Programming Language

* Python 3.9

Libraries

* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras
* Imbalanced-learn
* SHAP
* Matplotlib
* Seaborn

Development Tools

* Visual Studio Code
* Google Colab

---

# Hardware Used

* CPU: Intel Core i5-11320H
* RAM: 16GB DDR4
* Storage: 512GB SSD
* OS: Windows 11 64-bit

The entire pipeline was executed **without GPU acceleration**.

---

# Future Work

Potential extensions of this research include:

* Integration of **ultrasound image data**
* Development of **multimodal PCOS detection models**
* Deployment as a **web-based clinical decision support system**
* Mobile application for **early screening**
* Real-time patient monitoring and predictive analytics

---

# Citation

If you use this work in your research, please cite the associated paper.

```
Early Detection and Prediction of Polycystic Ovary Syndrome Using an Explainable Deep Learning Ensemble
```

---

# License

This project is intended for **academic and research purposes**.

---
