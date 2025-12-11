# NBA Triple-Double Prediction
## Introduction

Triple-doubles are rare events in the NBA, making them a challenging prediction problem due to heavy class imbalance. This project uses historical player box score data to build models capable of estimating triple-double probability based on recent player performance and game context.

The analysis includes:

Creating a binary target variable for triple-double occurrences

Generating rolling average features using recent player statistics

One-hot encoding game context variables such as opponent and game type

Training Logistic Regression and XGBoost classification models

Addressing class imbalance through undersampling and class-weighting

Optimizing decision thresholds to improve F1 performance

Evaluating and comparing models on a held-out test set

The project is implemented entirely within the accompanying Jupyter notebook.

## Data Source

The dataset used in this project is available on Kaggle:

NBA Traditional Box Score Dataset
https://www.kaggle.com/datasets/szymonjwiak/nba-traditional

**Due to licensing and size restrictions, the full dataset is not included in this repository.**
To reproduce the results:

Download the dataset from Kaggle.

Place the file in the data/ directory as traditional.csv.

A small sample dataset may optionally be included for demonstration purposes.

Examples
Logistic Regression

A Logistic Regression model is trained with standardized features and class weighting to compensate for the extreme imbalance between triple-double and non–triple-double samples. Performance is evaluated using both the default probability threshold and an F1-optimized threshold.

XGBoost Classification

An XGBoost model is trained with parameter tuning using grid search and K-fold cross-validation. The model incorporates imbalance-handling strategies such as scaled positive class weighting. Threshold tuning is again performed to improve F1 score.

These examples illustrate the effect of feature engineering, imbalance handling, and threshold selection on prediction performance.

## Methodology
Label Construction

A triple-double is defined as achieving at least 10 in three or more of the following categories:

Points

Rebounds

Assists

Steals

Blocks

A binary target variable TRIPLE-DOUBLE is constructed using this definition.

### Feature Engineering

**Rolling Averages (Last 5 Games)**
Rolling means (shifted by one game to avoid target leakage) are computed per player for:
PTS, REB, AST, STL, BLK, and MIN.

**Game Context Variables**

HOME indicator

Game type one-hot encoding (TYPE_PLAYOFF, TYPE_REGULAR)

Opponent one-hot encoding (OPP_*)

**Final Feature Set**
A feature matrix containing rolling averages, context variables, and opponent encodings.

### Handling Class Imbalance

Undersampling of the majority class (non–triple-double games)

Class weighting for Logistic Regression

Positive class reweighting via scale_pos_weight for XGBoost

Threshold tuning to optimize F1 score
