import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


######################################################################
# Function Name : BreastCancerVotingHard
# Description   : Classify breast cancer using Hard Voting Classifier
# Author        : Pranav Avinash Narkhede
######################################################################

def BreastCancerVotingHard(DataPath):

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
    # Step 4 : Feature Scaling
    ##################################################################

    print("\n", Border)
    print("Step 4 : Feature Scaling")
    print(Border)

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Calculate mean and standard deviation using training data
    # Then scale the training features
    X_train_scaled = scaler.fit_transform(X_train)

    # Scale testing features using training data statistics
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed successfully")

    ##################################################################
    # Step 5 : Create Individual Models
    ##################################################################

    print("\n", Border)
    print("Step 5 : Create Individual Models")
    print(Border)

    # Create a Logistic Regression model
    model_log = LogisticRegression(max_iter=1000)

    # Create a Decision Tree model
    model_det = DecisionTreeClassifier(random_state=42)

    # Create a KNN model with 5 nearest neighbors
    model_knn = KNeighborsClassifier(n_neighbors=5)

    print("Logistic Regression model created")
    print("Decision Tree model created")
    print("KNN model created")

    ##################################################################
    # Step 6 : Create the Hard Voting Classifier
    ##################################################################

    print("\n", Border)
    print("Step 6 : Create Hard Voting Classifier")
    print(Border)

    # Combine the three individual models using Voting Classifier
    # Hard voting selects the class receiving the majority of votes

    model = VotingClassifier(
        estimators=[
            ('logistic', model_log),
            ('decision_tree', model_det),
            ('knn', model_knn)
        ],
        voting='hard'
    )

    print("Hard Voting Classifier created successfully")

    ##################################################################
    # Step 7 : Train the model
    ##################################################################

    print("\n", Border)
    print("Step 7 : Model Training")
    print(Border)

    # Train all individual models using the training data
    model.fit(X_train_scaled, Y_train)

    print("Model training completed successfully")

    ##################################################################
    # Step 8 : Test the model
    ##################################################################

    print("\n", Border)
    print("Step 8 : Model Testing")
    print(Border)

    # Predict the target values using testing data
    Y_pred = model.predict(X_test_scaled)

    print("Model testing completed successfully")

    ##################################################################
    # Step 9 : Evaluate the model
    ##################################################################

    print("\n", Border)
    print("Step 9 : Model Evaluation")
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

    # Start breast cancer classification using Hard Voting
    BreastCancerVotingHard("dataset/breast_cancer.csv")


if __name__ == "__main__":
    main()