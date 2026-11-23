# Breast Cancer Classification

## Overview

This project uses Machine Learning classification algorithms to predict whether a breast cancer tumor is **Malignant** or **Benign**.

The project uses the Breast Cancer dataset and different Machine Learning algorithms to understand and compare classification techniques.

## Algorithms Implemented

The following five algorithms are implemented:

### 1. Random Forest
- Uses multiple Decision Trees to make predictions.
- Combines their predictions to produce the final result.
- Helps improve prediction accuracy.

File: `breast_cancer.py`

### 2. Bagging
- Creates multiple models using different samples of the training data.
- Combines predictions from these models.
- Helps reduce overfitting and improve model stability.

File: `breast_cancer_bagging.py`

### 3. AdaBoost
- Combines multiple weak learners to create a stronger model.
- Gives more importance to incorrectly classified samples.
- Tries to improve performance in each iteration.

File: `breast_cancer_adaboost.py`

### 4. Hard Voting
- Combines predictions from Logistic Regression, Decision Tree, and KNN.
- Each model votes for a class.
- The class receiving the majority of votes becomes the final prediction.

File: `breast_cancer_voting_hard.py`

### 5. Soft Voting
- Combines Logistic Regression, Decision Tree, and KNN.
- Uses the predicted probabilities from each model.
- Selects the class with the highest average probability.

File: `breast_cancer_voting_soft.py`

## Dataset

Dataset location: `dataset/breast_cancer.csv`

- Input features: Breast cancer measurements.
- Target column: `target`
- Target values represent the two tumor classes.

Note: Verify the dataset's target encoding before interpreting `0` and `1` as Malignant or Benign.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib (if used for visualizations)

## Project Structure

```text
04-Breast-Cancer-Classification/
│
├── dataset/
│   └── breast_cancer.csv
│
├── breast_cancer.py
├── breast_cancer_bagging.py
├── breast_cancer_adaboost.py
├── breast_cancer_voting_hard.py
├── breast_cancer_voting_soft.py
│
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib
```

## How to Run

Run the required Python file from the project directory.

```bash
python breast_cancer.py
```

For other algorithms:

```bash
python breast_cancer_bagging.py
python breast_cancer_adaboost.py
python breast_cancer_voting_hard.py
python breast_cancer_voting_soft.py
```

## Evaluation

Each program evaluates its model using:

- Accuracy Score
- Confusion Matrix

## Learning Outcomes

Through this project, I learned:

- Random Forest classification.
- Bagging and Boosting techniques.
- Hard Voting and Soft Voting.
- Ensemble Learning.
- Model training and prediction.
- Model evaluation using accuracy and confusion matrix.
- Feature scaling for models that require it.