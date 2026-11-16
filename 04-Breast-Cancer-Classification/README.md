# Breast Cancer Classification Using Random Forest

## Overview

Breast Cancer Classification is a Machine Learning project that uses the Random Forest Classifier to classify breast cancer cases based on different medical features.

Random Forest combines multiple Decision Trees to make predictions and improve classification performance.

## Objectives

- Understand the working of the Random Forest algorithm.
- Load and analyze a dataset using Pandas.
- Separate input features and target variables.
- Split the dataset into training and testing sets.
- Train a Random Forest classification model.
- Evaluate model performance using accuracy and confusion matrix.

## Technologies Used

- **Python** – Programming language
- **Pandas** – Dataset loading and analysis
- **Scikit-learn** – Model training, testing, and evaluation

## Dataset

The project uses `breast_cancer.csv`.

The dataset contains medical features used to classify breast cancer cases.

### Input Features
The dataset contains multiple numerical features describing characteristics of cell nuclei.

### Target Variable
- **`target`** – Represents the breast cancer classification.

The target values are:
- `0` – Malignant
- `1` – Benign

## Project Workflow

1. **Dataset Loading:** Load the CSV dataset using Pandas.
2. **Feature Selection:** Separate independent features and the target variable.
3. **Train-Test Split:** Divide the dataset into 80% training and 20% testing data.
4. **Model Creation:** Create a Random Forest Classifier.
5. **Model Training:** Train the model using training data.
6. **Model Testing:** Predict results using testing data.
7. **Model Evaluation:** Calculate accuracy and display the confusion matrix.

## Algorithm Used

### Random Forest Classifier

Random Forest is a supervised Machine Learning algorithm used for classification and regression.

It creates multiple Decision Trees and combines their predictions to produce the final result.

In this project, the Random Forest model uses 10 Decision Trees to classify breast cancer cases.

## Project Structure

```text
04-Breast-Cancer-Classification/
│
├── README.md
├── breast_cancer.py
│
└── dataset/
    └── breast_cancer.csv
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/pranavavinashnarkhede/Machine-Learning-Projects.git
```

### 2. Navigate to the Project Directory

```bash
cd Machine-Learning-Projects/04-Breast-Cancer-Classification
```

### 3. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 4. Run the Program

```bash
python breast_cancer.py
```

## Model Evaluation

The model performance is evaluated using:

- **Accuracy:** Measures the percentage of correctly classified predictions.
- **Confusion Matrix:** Displays the number of correct and incorrect predictions for each class.

## Learning Outcomes

Through this project, I learned:

- How Random Forest works.
- How multiple Decision Trees contribute to an ensemble model.
- How to split a dataset into training and testing sets.
- How to train and test a classification model.
- How to evaluate model performance using accuracy and confusion matrix.

## Author

**Pranav Avinash Narkhede**

GitHub: [pranavavinashnarkhede](https://github.com/pranavavinashnarkhede)