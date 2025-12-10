# NBA-Triple-Double-Prediction
A basic machine learning project that predicts the likelihood of an NBA player achieving a triple-double by combining exploratory data analysis, feature engineering, and multiple classification models to uncover the key factors driving elite in-game performance.
## Project Overview

This project focuses on predicting whether an NBA player will achieve a triple-double in a given game. Using historical player game logs, I performed exploratory data analysis, cleaned and transformed the dataset, and applied classification models to estimate triple-double likelihood.

The project explores which game statistics most strongly influence triple-double performance and how different models handle the rarity of triple-doubles.

### Skills Demonstrated

Machine Learning Classification

Feature Engineering

Handling Imbalanced Datasets

Exploratory Data Analysis (EDA)

Model Evaluation (Precision, Recall, F1, AUC)

Data Cleaning & Preprocessing

Visualization & Insight Communication

### Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook

# Project Workflow
1. Data Collection & Cleaning

Collected and standardized NBA player game logs

Cleaned missing values and normalized key statistical fields

Added engineered features such as recent performance trends and usage-based metrics

2. Exploratory Data Analysis

Visualized distributions of common player statistics

Compared triple-double vs. non–triple-double game profiles

Identified patterns in assists, rebounds, minutes played, and efficiency

3. Feature Engineering

Examples of engineered predictors:

Rolling averages of player stats

Rebound + assist ratios

Usage rate and efficiency metrics

Momentum-based features

4. Modeling

Experimented with multiple classification models:

Logistic Regression

Random Forest

Decision Tree

(Add more if applicable)

Approaches for class imbalance:

Class weighting

Oversampling (if used)

5. Evaluation Metrics

Confusion Matrix

Precision & Recall

F1-Score

ROC-AUC

### Key Insights

Triple-doubles occur in less than 1% of NBA player games, making prediction challenging and requiring imbalance-aware methods.

Minutes played, rebounds, assists, and usage metrics were among the strongest predictors.

Models using class weighting significantly improved recall (catching more true triple-double cases).
