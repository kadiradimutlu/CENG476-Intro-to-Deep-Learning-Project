# Fake News Detection using Ensemble Deep Learning

### CENG 476 - Deep Learning Project
**Group Members:**
* **Kadir ADIMUTLU** (210444003)
* **Muhammed YILDIZ** (210444052)
* **Doğan ÇAKIR** (210444075)

---

### Project Overview
This project implements a robust Deep Learning system to classify news articles as "Real" or "Fake" using the ISOT dataset. We compare traditional machine learning baselines against advanced deep learning architectures and a final ensemble.

### Architectures Implemented
We implemented and compared four distinct approaches:
1.  **Baseline Model:** Logistic Regression with TF-IDF vectorization (Benchmark).
2.  **Bi-Directional LSTM:** A Recurrent Neural Network with explicit experiments on **Regularization** (Dropout & L2) and **Batch Normalization**.
3.  **BERT:** Transfer Learning using the `bert-base-uncased` transformer model.
4.  **Ensemble Model:** A weighted average of LSTM and BERT predictions.

---

### Final Results
The ensemble model achieved near-perfect classification performance, demonstrating that combining sequence modeling (LSTM) with contextual attention (BERT) yields the most robust results.

| Model | Accuracy | F1-Score | AUC Score |
| :--- | :--- | :--- | :--- |
| **Baseline (LogReg)** | 98.81% | 0.9876 | 0.9984 |
| **LSTM (Final)** | 99.26% | 0.9922 | 0.9996 |
| **BERT** | 99.93% | 0.9992 | 0.9999 |
| **Ensemble** | **99.90%** | **0.9989** | **1.0000** |

*Note: While BERT slightly edged out the Ensemble in raw accuracy by 2 samples, the Ensemble achieved a perfect **1.00 AUC**, indicating superior stability and class separation.*

---

### Key Experiments & Rubric Requirements
* **Ablation Study:** We compared a "Low Regularization" LSTM (Dropout 0.2) against a "High Regularization" LSTM (Dropout 0.5 + L2 Weight Decay). The high regularization model generalized better (99.26% vs 99.14%).
* **Optimization:** Utilized `ReduceLROnPlateau` scheduler and `EarlyStopping` to prevent overfitting.
* **Evaluation:** Comprehensive reporting including Confusion Matrix, ROC Curves, and Per-Class metrics.

---

### Dataset
We used the **Fake and Real News Dataset** from Kaggle.
*Due to file size limits, the dataset is not included in this repository.*

**Download Instructions:**
1.  Download `True.csv` and `Fake.csv` from the link below.
2.  Place them in the same directory as the notebook (or upload to the Google Colab session).

🔗 **[Dataset Link (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)**

---

### How to Run
1.  Open the `.ipynb` file in **Google Colab**.
2.  Upload `Fake.csv` and `True.csv` to the files section.
3.  Run all cells. (The code handles library installation and data preprocessing automatically).
