# Human Activity Recognition (HAR) Using PCA and LDA

This project explores and compares dimensionality reduction techniques—specifically **Principal Component Analysis (PCA)** as an unsupervised method and **Linear Discriminant Analysis (LDA)** as a supervised method—on the Human Activity Recognition dataset.

---

## 📌 Project Overview
The main objective is to reduce high-dimensional sensor data (561 features) from smartphones and evaluate how feature representation affects the classification performance of a Logistic Regression model.

---

## 🔍 Methodology & Steps
1. **Data Preparation & Standardization:** Loaded smartphone sensor data, combined training and testing splits, and applied `StandardScaler` (Z-score normalization) to ensure all features contribute equally.
2. **Unsupervised Reduction (PCA):** 
   - Retained $\sim$63 components to preserve at least 90% of the total variance.
3. **Supervised Reduction (LDA):** 
   - Reduced the feature space to 5 dimensions ($C - 1$, where $C = 6$ activity classes) by maximizing class separability using activity labels.

---

## 🤖 Results and Performance
* **PCA + Logistic Regression (63 components):**
  * **Accuracy:** 92.22% | **Precision:** 92.26% | **Recall:** 92.22% | **F1-Score:** 92.22%
* **LDA + Logistic Regression (5 components):**
  * **Accuracy:** 96.50% | **Precision:** 96.60% | **Recall:** 96.50% | **F1-Score:** 96.50%

---

## 💡 Conclusion
While PCA significantly reduced dimensionality while retaining data variance, **LDA** clearly outperformed PCA by achieving higher classification accuracy with a much smaller feature space (5 components vs. 63). This demonstrates that supervised dimensionality reduction is highly effective when class labels are available.

---

## 🛠️ Tech Stack
* Python, Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn (`PCA`, `LinearDiscriminantAnalysis`, `StandardScaler`, `LogisticRegression`, metrics)

