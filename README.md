# Credit Card Fraud Detection — Capstone Project

**Author:** Dennis B. 
**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🚀 Live Demo

An interactive fraud detection demo built from this project is available here:

👉 [fraud_demo](https://github.com/Denxio/fraud_demo)

---

## Project Overview

This project demonstrates an end-to-end machine learning lifecycle applied to credit card fraud detection — a real-world, industry-relevant imbalanced classification problem. It covers:

- Problem framing & business cost analysis
- Data preprocessing & exploratory analysis
- Model training & evaluation (Logistic Regression, Random Forest, XGBoost)
- Handling class imbalance via **SMOTE** and **class weights**
- Interpretability via **SHAP analysis**
- Cross-validation, overfitting diagnostics, and fairness analysis
- Production recommendations

---

## Key Findings

- **Best metric:** AUPRC (Area Under Precision-Recall Curve) — far more meaningful than accuracy on severely imbalanced data (~0.17% fraud rate).
- **Best model:** XGBoost + Class Weights (AUPRC = 0.82 on test set) with a decision threshold of **0.104**, tuned using a business cost ratio of $120 (false negative) vs $8 (false positive).
- **Top features:** V14, V4, and V12 (SHAP global ranking); V14, V10, and V12 locally for individual fraud transactions.
- **Isolation Forest** performs well as an unsupervised baseline and is recommended as a complementary second-opinion layer in production.

---

## Files

| File | Description |
|------|-------------|
| `Pillar5_Capstone_Project.ipynb` | Full analysis notebook (preprocessing → modelling → evaluation) |
| `capstone_technical_presentation_slides.html` | Technical presentation slides |

---

## Setup & Running the Notebook

### 1. Get the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root. The notebook expects the file at:

```
raw_creditcard.csv
```

> **Note:** The notebook was originally developed in Google Colab. If running locally, update the file paths in cells that reference `/content/raw_creditcard.csv` to point to your local copy.

### 2. Install dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

### 3. Run the notebook

```bash
jupyter notebook Pillar5_Capstone_Project.ipynb
```

---

## Dependencies

- Python 3.x
- numpy, pandas, scipy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- xgboost
- shap

---

## Production Recommendation

Deploy **XGBoost + Class Weights** with a decision threshold of `0.104`. Prerequisites before production:

1. Full demographic fairness audit using actual cardholder data
2. Stronger regularisation to reduce the train/test AUPRC gap (~0.18)
3. Temporal train/test split to validate on future transactions
4. Legal & compliance review (ECOA, EU AI Act)

---

## License

This project is for educational purposes.
# AIM
