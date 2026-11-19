import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


######################################################################
# Function Name : BreastCancerBagging
# Description   : Classify breast cancer using Bagging Classifier
# Author        : Pranav Avinash Narkhede
######################################################################

def BreastCancerBagging(DataPath):

    Border = "-" * 70

    ##################################################################
    # Step 1 : Load the dataset
    ##################################################################

    print(Border)
    print("Step 1 : Dataset Loading")
    print(Border)

    # Load the CSV dataset using Pandas
    df = pd.read_csv(DataPath)

    print("Dataset loaded successfully")
    print("Shape of dataset : ", df.shape)

    # Display the first five records
    print("\nFirst five records : ")
    print(df.head())

    ##################################################################
    # Step 2 : Separate input features and target variable
    ##################################################################

    print("\n", Border)
    print("Step 2 : Separate Features and Labels")
    print(Border)

    # X contains all input features except the target column
    X = df.drop("target", axis=1)

    # Y contains the target column that we want to predict
    Y = df["target"]

    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    print("Input Features : ", X.columns.tolist())
    print("Target Variable : target")

    ##################################################################
    # Step 3 : Split the dataset into training and testing data
    ##################################################################

    print("\n", Border)
    print("Step 3 : Train-Test Split")
    print(Border)

    # Divide the dataset into 80% training and 20% testing data
    # random_state ensures the same split every time
    # stratify maintains class proportions in both datasets

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("Training and testing data created successfully")

    print("X_train Shape : ", X_train.shape)
    print("X_test Shape  : ", X_test.shape)

    print("Y_train Shape : ", Y_train.shape)
    print("Y_test Shape  : ", Y_test.shape)

    ##################################################################
    # Step 4 : Create the Base Model
    ##################################################################

    print("\n", Border)
    print("Step 4 : Create Base Model")
    print(Border)

    # Create a Decision Tree classifier
    # This model will be used as the base estimator for Bagging

    base_model = DecisionTreeClassifier(random_state=42)

    print("Base Decision Tree model created successfully")

    ##################################################################
    # Step 5 : Create the Bagging Model
    ##################################################################

    print("\n", Border)
    print("Step 5 : Create Bagging Model")
    print(Border)

    # Create a Bagging Classifier
    # n_estimators=10 means 10 base models will be created
    # Bagging combines their predictions to produce the final result

    model = BaggingClassifier(
        estimator=base_model,
        n_estimators=10,
        random_state=42
    )

    print("Bagging model created successfully")

    ##################################################################
    # Step 6 : Train the model
    ##################################################################

    print("\n", Border)
    print("Step 6 : Model Training")
    print(Border)

    # Train the Bagging model using training data
    model.fit(X_train, Y_train)

    print("Model training completed successfully")

    ##################################################################
    # Step 7 : Test the model
    ##################################################################

    print("\n", Border)
    print("Step 7 : Model Testing")
    print(Border)

    # Predict the target values using testing data
    Y_pred = model.predict(X_test)

    print("Model testing completed successfully")

    ##################################################################
    # Step 8 : Evaluate the model
    ##################################################################

    print("\n", Border)
    print("Step 8 : Model Evaluation")
    print(Border)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(Y_test, Y_pred)

    print("Model Accuracy : {:.2f}%".format(accuracy * 100))

    # Display the confusion matrix
    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_test, Y_pred))


######################################################################
# Function Name : main
# Description   : Starting point of the program
######################################################################

def main():

    # Start breast cancer classification using Bagging
    BreastCancerBagging("dataset/breast_cancer.csv")


if __name__ == "__main__":
    main()