import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


#####################################################################
# Function Name : MarvellousClassifier
# Description   : Train and test a KNN model using the Wine dataset
# Author        : Pranav Avinash Narkhede
#####################################################################

def MarvellousClassifier(DataPath):

    border = "-" * 50

    #################################################################
    # Step 1 : Load the dataset
    #################################################################

    print(border)
    print("Step 1 : Load the dataset from CSV file")
    print(border)

    # Read the CSV file and store its data in a DataFrame
    df = pd.read_csv(DataPath)

    print(border)
    print("Some entries from dataset : ")

    # Display the first five rows of the dataset
    print(df.head())

    print(border)

    #################################################################
    # Step 2 : Clean the dataset
    #################################################################

    print(border)
    print("Step 2 : Clean the dataset")
    print(border)

    # Remove rows containing missing values
    df.dropna(inplace=True)

    # Display information about the dataset
    print("Shape of dataset : ", df.shape)

    # Display the total number of rows
    print("Total records : ", df.shape[0])

    # Display the total number of columns
    print("Total columns : ", df.shape[1])

    print(border)

    #################################################################
    # Step 3 : Separate input and output variables
    #################################################################

    print(border)
    print("Step 3 : Separate independent and dependent variables")
    print(border)

    # X contains all columns except the target column 'Class'
    X = df.drop(columns=['Class'])

    # Y contains the target column that we want to predict
    Y = df['Class']

    # Display the dimensions of input and output data
    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    print(border)

    # Display the names of input and output columns
    print("Input columns : ", X.columns.tolist())
    print("Output columns : Class")

    #################################################################
    # Step 4 : Split the dataset
    #################################################################

    print(border)
    print("Step 4 : Split the dataset for training and testing")
    print(border)

    # Divide the dataset into 50% training and 50% testing data
    # random_state=42 gives the same split every time
    # stratify=Y maintains the class proportions in both sets

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42,
        stratify=Y
    )

    print(border)
    print("Details of training and testing data")

    # Display the dimensions of training and testing data
    print("Shape of X_train : ", X_train.shape)
    print("Shape of X_test : ", X_test.shape)

    print("Shape of Y_train : ", Y_train.shape)
    print("Shape of Y_test : ", Y_test.shape)

    print(border)

    #################################################################
    # Step 5 : Feature Scaling
    #################################################################

    print(border)
    print("Step 5 : Feature Scaling")
    print(border)

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Calculate mean and standard deviation from training data
    # Then scale the training features
    X_train_scaled = scaler.fit_transform(X_train)

    # Scale testing features using the training data statistics
    X_test_scaled = scaler.transform(X_test)

    print("Feature Scaling done")

    print(border)

    #################################################################
    # Step 6 : Hyperparameter Tuning
    #################################################################

    # Store accuracy scores for different values of K
    accuracy_scores = []

    # Try K values from 1 to 20
    K_values = range(1, 21)

    # Train and test the KNN model for each K value
    for k in K_values:

        # Create a KNN classifier with K nearest neighbors
        model = KNeighborsClassifier(n_neighbors=k)

        # Train the model using scaled training data
        model = model.fit(X_train_scaled, Y_train)

        # Predict classes for the testing data
        Y_pred = model.predict(X_test_scaled)

        # Calculate the accuracy of the current model
        accuracy = accuracy_score(Y_test, Y_pred)

        # Store the accuracy for the current K value
        accuracy_scores.append(accuracy)

    print("Accuracy report : ")

    # Display accuracy scores for all K values
    for no in accuracy_scores:
        print(no)

    print(border)

    #################################################################
    # Step 7 : Data Visualization
    #################################################################

    print(border)
    print("Graphical representation")
    print(border)

    # Create a figure with width 8 and height 5
    plt.figure(figsize=(8, 5))

    # Plot K values on the X-axis and accuracy on the Y-axis
    plt.plot(K_values, accuracy_scores, marker="o")

    # Add graph title and axis labels
    plt.title("K values Vs Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")

    # Display grid lines
    plt.grid(True)

    # Display every K value on the X-axis
    plt.xticks(list(K_values))

    # Display the graph
    plt.show()


#####################################################################
# Function Name : main
# Description   : Starting point of the program
# Author        : Pranav Avinash Narkhede
#####################################################################

def main():

    # Start the classification process using the Wine dataset
    MarvellousClassifier("dataset/WinePredictor.csv")


if __name__ == "__main__":
    main()