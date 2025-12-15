# NBA Triple-Double Prediction

This repository contains an end-to-end machine learning project designed to predict the likelihood of an NBA player achieving a triple-double in a given game. The project includes data preprocessing, feature engineering, model development, imbalance handling, threshold optimization, and evaluation. The goal is to explore how effectively machine learning models can identify rare, high-performance outcomes in professional basketball.

---

- [Data Source](#data-source)
- [Repository structure](#repository-structure)
- [Installation method](#installation)
- [Methodolody](#methodology)
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
├── images/ (none included yet)
|
│
|── README.md
```

---

# Installation

1. Clone the repository:
```
git clone https://github.com/<your-username>/NBA-Triple-Double-Prediction.git
cd NBA-Triple-Double-Prediction
```

2. Install dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn
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
## Models used

### Logistic Regression  
A Logistic Regression model is trained with standardized features and class weighting to compensate for the extreme imbalance between triple-double and non–triple-double samples. Performance is evaluated using both the default probability threshold and an F1-optimized threshold.

### XGBoost Classification  
An XGBoost model is trained with parameter tuning using grid search and K-fold cross-validation. The model incorporates imbalance-handling strategies such as scaled positive class weighting. Threshold tuning is again performed to improve F1 score.

These examples illustrate the effect of feature engineering, imbalance handling, and threshold selection on prediction performance.

---

## Methodology

### 1. Data Preprocessing
- Sorted player game histories chronologically
- Removed data leakage by using **only prior game information**
- Encoded categorical variables
- Handled missing values

### 2. Feature Engineering
- Rolling and lag-based performance features
- Game-type indicators (regular season vs. playoffs)
- Player-specific historical trends

### 3. Modeling
- Logistic Regression (baseline model)
- Additional classification models explored for performance comparison

### 4. Evaluation
- Precision, recall, and F1-score
- Confusion matrix
- ROC–AUC

Special attention was given to **class imbalance**, as triple-doubles are relatively rare events.
---
## Tools & Technologies

- **Python**
- **Pandas/NumPy**
- **Scikit-learn**
- Matplotlib / Seaborn**
- **Jupyter Notebook**

---

# Key Insights

- Logistic Regression performs extremely well on the majority class but has limited ability to detect rare triple-double events without threshold tuning.
- Threshold tuning significantly increases recall for both models, demonstrating the importance of adjusting decision boundaries in imbalanced classification.
- XGBoost outperforms Logistic Regression in recall and F1 when using the optimized threshold, showing its ability to model more complex relationships.
- Even with advanced techniques, predicting triple-doubles remains difficult due to extreme class imbalance and variability in player performance.
- Rolling performance features and opponent/game context information help the model identify meaningful patterns, but prediction remains uncertain due to rarity.

---

# Future Work

- Expand modeling to include Random Forest, LightGBM, or Neural Networks  
- Explore temporal models such as LSTM or transformer-based architectures  
- Add player tracking and possession-level features  
- Deploy a probability dashboard for triple-double prediction  
- Experiment with oversampling methods such as SMOTE  

---
# License
This project is licensed under the **MIT License**

