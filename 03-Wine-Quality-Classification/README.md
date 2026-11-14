# Wine Classification Using KNN

## Overview
Wine Classification is a Machine Learning project that uses the K-Nearest Neighbors (KNN) algorithm to classify wines into different categories based on their chemical properties.

## Objectives
- Understand the KNN classification algorithm.
- Load and analyze a dataset using Pandas.
- Perform basic data cleaning.
- Apply feature scaling using StandardScaler.
- Train and test a KNN classification model.
- Understand the effect of different K values on model accuracy.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
The project uses `WinePredictor.csv`.

The dataset contains different wine-related features and a target column named `Class`.

- **Input Features:** Chemical properties of wine.
- **Target Variable:** `Class` (wine category).

## Project Workflow
1. **Dataset Loading:** Load the CSV dataset using Pandas.
2. **Data Cleaning:** Remove rows containing missing values.
3. **Feature Selection:** Separate independent and dependent variables.
4. **Train-Test Split:** Divide the dataset into training and testing sets.
5. **Feature Scaling:** Scale input features using StandardScaler.
6. **Model Training and Testing:** Train KNN models using different K values.
7. **Accuracy Calculation:** Calculate accuracy for each K value.
8. **Visualization:** Plot a graph showing K values against accuracy.

## Algorithm Used

### K-Nearest Neighbors (KNN)

KNN is a supervised Machine Learning algorithm used for classification. It predicts the class of a new data point by finding its nearest K data points and selecting the most common class among them.

## Project Structure
```text
03-Wine-Classification-KNN/
├── README.md
├── wine_classification.py
├── requirements.txt
└── dataset/
    └── WinePredictor.csv
```

## Installation and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/pranavavinashnarkhede/Machine-Learning-Projects.git
```

### 2. Navigate to the Project Directory
```bash
cd Machine-Learning-Projects/03-Wine-Classification-KNN
```

### 3. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 4. Run the Program
```bash
python wine_classification.py
```

## Learning Outcomes
- Understood the working of the KNN algorithm.
- Learned how to perform basic data cleaning.
- Understood the importance of feature scaling.
- Learned how to split data into training and testing sets.
- Explored different K values and their effect on accuracy.
- Visualized model accuracy using Matplotlib.

## Author
**Pranav Avinash Narkhede**

GitHub: https://github.com/pranavavinashnarkhede