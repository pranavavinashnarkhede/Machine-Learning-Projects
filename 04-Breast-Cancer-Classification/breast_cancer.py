import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def BreastCancerClassifier(DataPath):

    Border = "-" * 70

    ######################################################################
    # Step 1 : Load the dataset
    ######################################################################

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

    ######################################################################
    # Step 2 : Separate input features and target variable
    ######################################################################

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

    ######################################################################
    # Step 3 : Split the dataset into training and testing data
    ######################################################################

    print("\n", Border)
    print("Step 3 : Train-Test Split")
    print(Border)

    # Divide the dataset into 80% training and 20% testing data
    # random_state ensures reproducible results
    # stratify maintains the class distribution in both sets

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

    ######################################################################
    # Step 4 : Create the Random Forest model
    ######################################################################

    print("\n", Border)
    print("Step 4 : Model Creation")
    print(Border)

    # Create a Random Forest classifier
    # n_estimators represents the number of decision trees

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    print("Random Forest model created successfully")

    ######################################################################
    # Step 5 : Train the model
    ######################################################################

    print("\n", Border)
    print("Step 5 : Model Training")
    print(Border)

    # Train the Random Forest model using training data
    model.fit(X_train, Y_train)

    print("Model training completed successfully")

    ######################################################################
    # Step 6 : Test the model
    ######################################################################

    print("\n", Border)
    print("Step 6 : Model Testing")
    print(Border)

    # Predict the target values using testing data
    Y_pred = model.predict(X_test)

    print("Model testing completed successfully")

    ######################################################################
    # Step 7 : Evaluate the model
    ######################################################################

    print("\n", Border)
    print("Step 7 : Model Evaluation")
    print(Border)

    # Calculate model accuracy using actual and predicted values
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

    # Start breast cancer classification using the dataset
    BreastCancerClassifier("dataset/breast_cancer.csv")


if __name__ == "__main__":
    main()