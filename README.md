# NBA Triple-Double Prediction

This repository contains an end-to-end machine learning project designed to predict the likelihood of an NBA player achieving a triple-double in a given game. The project includes data preprocessing, feature engineering, model development, imbalance handling, threshold optimization, and evaluation. The goal is to explore how effectively machine learning models can identify rare, high-performance outcomes in professional basketball.

---

## Introduction

Triple-doubles are rare events in the NBA, making them a challenging prediction problem due to heavy class imbalance. This project uses historical player box score data to build models capable of estimating triple-double probability based on recent player performance and game context.

The analysis includes:

- Creating a binary target variable for triple-double occurrences  
- Generating rolling average features using recent player statistics  
- One-hot encoding game context variables such as opponent and game type  
- Training Logistic Regression and XGBoost classification models  
- Addressing class imbalance through undersampling and class-weighting  
- Optimizing decision thresholds to improve F1 performance  
- Evaluating and comparing models on a held-out test set  

The project is implemented entirely within the accompanying Jupyter notebook.

---

## Data Source

The dataset used in this project is available on Kaggle:

NBA Traditional Box Score Dataset  
https://www.kaggle.com/datasets/szymonjwiak/nba-traditional

Due to licensing and size restrictions, the full dataset is not included in this repository.  
To reproduce the results:

1. Download the dataset from Kaggle.  
2. Place the file in the `data/` directory as `traditional.csv`.

---

## Examples

### Logistic Regression  
A Logistic Regression model is trained with standardized features and class weighting to compensate for the extreme imbalance between triple-double and non–triple-double samples. Performance is evaluated using both the default probability threshold and an F1-optimized threshold.

### XGBoost Classification  
An XGBoost model is trained with parameter tuning using grid search and K-fold cross-validation. The model incorporates imbalance-handling strategies such as scaled positive class weighting. Threshold tuning is again performed to improve F1 score.

These examples illustrate the effect of feature engineering, imbalance handling, and threshold selection on prediction performance.

---

## Methodology

### Label Construction  
A triple-double is defined as achieving at least 10 in three or more of the following categories:

- Points  
- Rebounds  
- Assists  
- Steals  
- Blocks  

A binary target variable `TRIPLE-DOUBLE` is constructed using this definition.

### Feature Engineering

1. **Rolling Averages (Last 5 Games)**  
   Rolling means (shifted by one game to avoid target leakage) are computed per player for:  
   `PTS`, `REB`, `AST`, `STL`, `BLK`, and `MIN`.

2. **Game Context Variables**  
   - `HOME` indicator  
   - Game type one-hot encoding (`TYPE_PLAYOFF`, `TYPE_REGULAR`)  
   - Opponent one-hot encoding (`OPP_*`)

3. **Final Feature Set**  
   A feature matrix containing rolling averages, context variables, and opponent encodings.

### Handling Class Imbalance

- Undersampling of the majority class (non–triple-double games)  
- Class weighting for Logistic Regression  
- Positive class reweighting via `scale_pos_weight` for XGBoost  
- Threshold tuning to optimize F1 score

---

# Results

Below are the actual results from the training and evaluation pipeline.

---

# Logistic Regression

### Cross-Validation Best Parameters
```
{'logreg__C': 0.001,
 'logreg__class_weight': None,
 'logreg__penalty': 'l2'}
```

### Best Precision on Undersampled Training Set
```
0.8425069124423963
```

---

## Logistic Regression – Evaluation on Full Test Set (Threshold = 0.5)

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     66398
           1       0.28      0.15      0.20       300

    accuracy                           0.99     66698
   macro avg       0.64      0.58      0.60     66698
weighted avg       0.99      0.99      0.99     66698
```

Confusion Matrix:
```
[[66281   117]
 [  254    46]]
```

---

## Logistic Regression – F1-Optimized Threshold

**Best Threshold:**
```
0.32
```

**Best F1 Score:**
```
0.2695187165775401
```

### Evaluation at F1-Optimized Threshold
```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99     66398
           1       0.20      0.42      0.27       300

    accuracy                           0.99     66698
   macro avg       0.60      0.71      0.63     66698
weighted avg       0.99      0.99      0.99     66698
```

Confusion Matrix:
```
[[65889   509]
 [  174   126]]
```

---

# XGBoost

### scale_pos_weight During Training:
```
221.8207126948775
```

### Cross-Validation Best Parameters
```
{
 'colsample_bytree': 1.0,
 'learning_rate': 0.01,
 'max_depth': 3,
 'n_estimators': 200,
 'scale_pos_weight': 1,
 'subsample': 0.8
}
```

### Best Accuracy in Cross-Validation
```
0.970121806177343
```

---

## XGBoost – Evaluation on Full Test Set (Threshold = 0.5)

```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     66398
           1       0.23      0.40      0.29       300

    accuracy                           0.99     66698
   macro avg       0.61      0.70      0.64     66698
weighted avg       0.99      0.99      0.99     66698
```

Confusion Matrix:
```
[[65996   402]
 [  179   121]]
```

Accuracy:
```
0.9706023633394179
```

---

## XGBoost – Evaluation at F1-Optimized Threshold

```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     66398
           1       0.14      0.60      0.23       300

    accuracy                           0.98     66698
   macro avg       0.57      0.79      0.61     66698
weighted avg       0.99      0.98      0.99     66698
```

Confusion Matrix:
```
[[65303  1095]
 [  120   180]]
```

---

# Key Insights

- Logistic Regression performs extremely well on the majority class but has limited ability to detect rare triple-double events without threshold tuning.
- Threshold tuning significantly increases recall for both models, demonstrating the importance of adjusting decision boundaries in imbalanced classification.
- XGBoost outperforms Logistic Regression in recall and F1 when using the optimized threshold, showing its ability to model more complex relationships.
- Even with advanced techniques, predicting triple-doubles remains difficult due to extreme class imbalance and variability in player performance.
- Rolling performance features and opponent/game context information help the model identify meaningful patterns, but prediction remains uncertain due to rarity.

---

# Repository Structure

```
NBA-Triple-Double-Prediction/
│
├── data/
│   └── traditional.csv (not included)
│
├── notebooks/
│   └── modeling.ipynb
│
├── images/
│   └── model_results.png (optional)
│
├── README.md
└── requirements.txt
```

---

# Installation

1. Clone the repository:
```
git clone https://github.com/<your-username>/NBA-Triple-Double-Prediction.git
cd NBA-Triple-Double-Prediction
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Add the dataset to the `data/` directory as:
```
data/traditional.csv
```

4. Run the notebook:
```
jupyter notebook notebooks/modeling.ipynb
```

---

# Future Work

- Expand modeling to include Random Forest, LightGBM, or Neural Networks  
- Explore temporal models such as LSTM or transformer-based architectures  
- Add player tracking and possession-level features  
- Deploy a probability dashboard for triple-double prediction  
- Experiment with oversampling methods such as SMOTE  

---

