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
https://www.kaggle.com/datasets

Due to licensing and size restrictions, the full dataset is not included in this repository.  
To reproduce the results:

1. Download the dataset from Kaggle.  
2. Place the file in the `data/` directory as `traditional.csv`.

A small sample dataset may optionally be included for demonstration purposes.

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

## Results

After model training and threshold optimization, the following evaluations are performed:

1. Logistic Regression (default threshold 0.5)  
2. Logistic Regression (F1-optimized threshold)  
3. XGBoost (default threshold 0.5)  
4. XGBoost (F1-optimized threshold)

Insert actual evaluation outputs here after running the notebook locally. For example:

```
Classification Report (Logistic Regression, threshold = 0.5)
Confusion Matrix (...)
Precision, Recall, F1 Score (...)

Classification Report (Logistic Regression, F1-optimized threshold = t*)
Confusion Matrix (...)
Precision, Recall, F1 Score (...)
```

Add confusion matrices, ROC curves, or any additional visualizations in the `images/` directory and reference them here.

---

## Repository Structure

```
NBA-Triple-Double-Prediction/
│
├── data/
│   └── traditional.csv              # (not included, see Data Source)
│
├── notebooks/
│   └── modeling.ipynb               # Main analysis notebook
│
├── images/
│   └── confusion_matrix.png
│   └── eda_visualization.png
│
├── README.md
└── requirements.txt
```

---

## Installation

To run this project locally:

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/NBA-Triple-Double-Prediction.git
   cd NBA-Triple-Double-Prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Add the dataset to the `data/` directory as `traditional.csv`.

4. Open and run the notebook:
   ```
   jupyter notebook notebooks/modeling.ipynb
   ```

---

## Future Work

- Additional models such as Random Forest, Gradient Boosting, or Neural Networks  
- Incorporation of more advanced player tracking data  
- Deployment of a triple-double probability dashboard  
- Inclusion of temporal models capturing player momentum over extended windows  

---

## Contact

Carlo Uy  
Email: carlo88uy@gmail.com  
LinkedIn: [add link]  
GitHub: [add link]

