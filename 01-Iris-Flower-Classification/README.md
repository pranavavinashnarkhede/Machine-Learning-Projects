# Iris Flower Classification Using Machine Learning

## 1. Overview

This project demonstrates how to build a Machine Learning classification model using Python and Scikit-learn to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

The model uses four flower measurements to predict the species using the Decision Tree Classification algorithm.

## 2. Objectives

* Understand and analyze the Iris dataset.
* Perform Exploratory Data Analysis (EDA).
* Visualize relationships between flower measurements.
* Train a Decision Tree classification model.
* Evaluate model performance using classification metrics.

## 3. Technologies Used

* **Python** — Programming language
* **Pandas** — Data loading and analysis
* **Matplotlib** — Data visualization
* **Scikit-learn** — Model training and evaluation

## 4. Dataset

The project uses `iris.csv`, which contains 150 records representing three Iris flower species.

### Features

| Feature      | Description                        |
| ------------ | ---------------------------------- |
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width  | Width of the sepal in centimeters  |
| Petal Length | Length of the petal in centimeters |
| Petal Width  | Width of the petal in centimeters  |

**Target Variable:** `species`

The target variable identifies the flower species: Setosa, Versicolor, or Virginica.

## 5. Project Workflow

The project follows these steps:

1. **Load Dataset:** Load the CSV file using Pandas.
2. **Exploratory Data Analysis:** Examine dataset dimensions, column names, missing values, class distribution, and statistical summaries.
3. **Feature Selection:** Select the four flower measurements as independent variables and `species` as the target variable.
4. **Data Visualization:** Create a scatter plot to visualize the relationship between petal length and petal width for each species.
5. **Train-Test Split:** Divide the dataset into 50% training data and 50% testing data.
6. **Model Creation:** Create a Decision Tree Classifier with `max_depth=5`.
7. **Model Training:** Train the classifier using the training dataset.
8. **Prediction:** Predict flower species for the testing dataset.
9. **Model Evaluation:** Evaluate predictions using accuracy, a confusion matrix, and a classification report.

## 6. Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree is a supervised Machine Learning algorithm that learns decision rules from training data to classify new observations.

In this project, the model uses the four flower measurements to predict the species.

The model is configured with:

```python
DecisionTreeClassifier(max_depth=5)
```

The `max_depth=5` parameter limits the maximum depth of the tree to help control model complexity.

## 7. Model Evaluation

The model's performance is evaluated using the following metrics:

* **Accuracy:** Measures the proportion of correct predictions.
* **Confusion Matrix:** Shows the number of correct and incorrect predictions for each class.
* **Precision:** Measures how many predictions for a particular class are correct.
* **Recall:** Measures how many actual instances of a class are correctly identified.
* **F1-Score:** Combines precision and recall into a single metric.

Actual evaluation results are generated when the Python script runs.

## 8. Project Structure

```text
01-Iris-Flower-Classification/
│
├── README.md
├── iris_classification.py
├── requirements.txt
│
└── dataset/
    └── iris.csv
```

## 9. Installation and Execution

### Prerequisites

* Python 3.x
* pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/pranavavinashnarkhede/Machine-Learning-Projects.git
```


### Step 2: Navigate to the Project Directory

```bash
cd Python-Machine-Learning-Case-Studies/01-Iris-Flower-Classification
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Python Script

```bash
python iris_classification.py
```

**Note:** Ensure that `iris.csv` is stored inside the `dataset` directory and that the dataset path in the Python script is set to `dataset/iris.csv`.

## 10. Key Learnings

Through this project, I explored:

* Loading and analyzing datasets using Pandas.
* Performing Exploratory Data Analysis.
* Visualizing data using Matplotlib.
* Separating independent and dependent variables.
* Splitting data into training and testing sets.
* Training a supervised Machine Learning model.
* Evaluating classification performance using Scikit-learn.

## 11. Future Improvements

* Compare Decision Tree performance with K-Nearest Neighbors and Logistic Regression.
* Experiment with different train-test split ratios.
* Tune model hyperparameters to improve generalization.
* Add feature importance visualization.
* Evaluate the model using cross-validation.

## 12. Conclusion

This case study demonstrates the fundamental workflow of a supervised Machine Learning classification problem, from dataset analysis and visualization to model training, prediction, and evaluation.

It provides a foundation for exploring more advanced Machine Learning algorithms and real-world classification problems.
