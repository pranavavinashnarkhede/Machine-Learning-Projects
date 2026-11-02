# Student Performance Prediction

## Overview

Student Performance Prediction is a beginner-friendly Machine Learning project that uses a Decision Tree Classifier to predict students' final results based on their academic and lifestyle-related features.

The model uses study hours, attendance, previous scores, completed assignments, and sleep hours to predict the final result.

## Objectives

* Understand the basic Machine Learning workflow.
* Load and analyze a dataset using Pandas.
* Visualize data using Matplotlib.
* Train a Decision Tree Classification model.
* Evaluate model performance using training and testing accuracy.

## Technologies Used

* **Python** — Programming language
* **Pandas** — Dataset loading and analysis
* **Matplotlib** — Data visualization
* **Scikit-learn** — Model training, dataset splitting, and evaluation

## Dataset

The project uses `student_performance_ml.csv`.

### Input Features

| Feature                | Description                     |
| ---------------------- | ------------------------------- |
| `StudyHours`           | Number of hours spent studying  |
| `Attendance`           | Student attendance              |
| `PreviousScore`        | Previous academic score         |
| `AssignmentsCompleted` | Number of completed assignments |
| `SleepHours`           | Number of hours spent sleeping  |

### Target Variable

* **`FinalResult`** — The result the model predicts.

The meanings of the target values (such as `0` and `1`) depend on how they are defined in the dataset.

## Project Workflow

1. **Dataset Loading:** Load the CSV file using Pandas.
2. **Data Analysis:** Check missing values, dataset dimensions, column names, and statistical information.
3. **Data Visualization:** Create a histogram to understand the distribution of students' study hours.
4. **Train-Test Split:** Divide the dataset into training and testing sets.
5. **Model Training:** Train a Decision Tree Classifier using the training data.
6. **Model Testing:** Generate predictions using the testing data.
7. **Model Evaluation:** Calculate training and testing accuracy.

## Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree Classifier learns patterns from the training data and uses a sequence of decision rules to predict the target variable.

It is suitable for learning the fundamentals of classification because its decision-making process can be visualized and understood.

## Project Structure

```text
02-Student-Performance-Prediction/
├── README.md
├── student_performance.py
└── dataset/
    └── student_performance_ml.csv
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/pranavavinashnarkhede/Machine-Learning-Projects.git
```

### 2. Navigate to the Project Directory

```bash
cd Machine-Learning-Projects/02-Student-Performance-Prediction
```

### 3. Install Dependencies

Run this command from the repository root, where `requirements.txt` is located:

```bash
pip install -r requirements.txt
```

### 4. Run the Program

Ensure that the dataset path in the Python script matches its location. Then run:

```bash
python student_performance.py
```

## Model Evaluation

The model's performance is evaluated using:

* **Training Accuracy:** Measures how accurately the model predicts results for the training data.
* **Testing Accuracy:** Measures how accurately the model predicts results for the testing data.

Actual accuracy values depend on the dataset, the train-test split, and the trained model.

## Learning Outcomes

Through this project, I learned:

* How to load and analyze datasets using Pandas.
* How to visualize data using Matplotlib.
* How to separate input features and target variables.
* How to split data into training and testing sets.
* How to train a Decision Tree Classifier.
* How to evaluate a classification model using accuracy.

## Future Improvements

* Compare different classification algorithms.
* Visualize the model's predictions.
* Explore additional evaluation metrics.
* Improve model performance through experimentation.

## Author

**Pranav Avinash Narkhede**

GitHub: [pranavavinashnarkhede](https://github.com/pranavavinashnarkhede)
