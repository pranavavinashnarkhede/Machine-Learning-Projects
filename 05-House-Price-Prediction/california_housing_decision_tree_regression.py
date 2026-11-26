import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


######################################################################
# Function Name : CaliforniaHousingDecisionTreeRegression
# Description   : Predict house prices using Decision Tree Regression
# Author        : Pranav Avinash Narkhede
######################################################################

def CaliforniaHousingDecisionTreeRegression(DataPath):

    Border = "-" * 70

    ##################################################################
    # Step 1 : Load the Dataset
    ##################################################################

    print(Border)
    print("Step 1 : Loading California Housing Dataset")
    print(Border)

    # Read the CSV file using Pandas.
    df = pd.read_csv(DataPath)

    print("Dataset Loaded Successfully")
    print("Dataset Shape :", df.shape)

    print("\nFirst Five Records:")
    print(df.head())

    ##################################################################
    # Step 2 : Separate Input and Output
    ##################################################################

    print("\n" + Border)
    print("Step 2 : Separating Input and Output")
    print(Border)

    # Use all eight input features for prediction.
    # The target column contains the house value to be predicted.
    X = df.drop("target", axis=1)
    Y = df["target"]

    print("Input Shape  :", X.shape)
    print("Output Shape :", Y.shape)

    print("\nInput Features:")
    print(X.columns.tolist())

    ##################################################################
    # Step 3 : Split the Dataset
    ##################################################################

    print("\n" + Border)
    print("Step 3 : Splitting Dataset into Training and Testing Data")
    print(Border)

    # Split the dataset into 80% training data and 20% testing data.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("Training Input Shape  :", X_train.shape)
    print("Testing Input Shape   :", X_test.shape)
    print("Training Output Shape :", Y_train.shape)
    print("Testing Output Shape  :", Y_test.shape)

    ##################################################################
    # Step 4 : Create the Decision Tree Regression Model
    ##################################################################

    print("\n" + Border)
    print("Step 4 : Creating Decision Tree Regression Model")
    print(Border)

    # Create an object of DecisionTreeRegressor.
    # The model learns decision rules by splitting the dataset
    # based on different input feature values.
    model = DecisionTreeRegressor(random_state=42)

    print("Decision Tree Regression Model Created Successfully")

    ##################################################################
    # Step 5 : Train the Model
    ##################################################################

    print("\n" + Border)
    print("Step 5 : Training the Model")
    print(Border)

    # Train the model using training input and output data.
    model.fit(X_train, Y_train)

    print("Model Training Completed Successfully")

    ##################################################################
    # Step 6 : Make Predictions
    ##################################################################

    print("\n" + Border)
    print("Step 6 : Making Predictions")
    print(Border)

    # Predict house values using testing input data.
    Y_pred = model.predict(X_test)

    print("Predictions Generated Successfully")

    print("\nActual Values:")
    print(Y_test.head())

    print("\nPredicted Values:")
    print(Y_pred[:5])

    ##################################################################
    # Step 7 : Evaluate the Model
    ##################################################################

    print("\n" + Border)
    print("Step 7 : Evaluating the Model")
    print(Border)

    # Calculate Mean Absolute Error.
    MAE = mean_absolute_error(Y_test, Y_pred)

    # Calculate Mean Squared Error.
    MSE = mean_squared_error(Y_test, Y_pred)

    # Calculate Root Mean Squared Error.
    RMSE = np.sqrt(MSE)

    # Calculate R2 Score.
    R2 = r2_score(Y_test, Y_pred)

    print("Mean Absolute Error :", MAE)
    print("Mean Squared Error  :", MSE)
    print("Root Mean Squared Error :", RMSE)
    print("R2 Score :", R2)

    ##################################################################
    # Step 8 : Display Model Information
    ##################################################################

    print("\n" + Border)
    print("Step 8 : Displaying Model Information")
    print(Border)

    # Display the maximum depth of the trained decision tree.
    print("Tree Depth :", model.get_depth())

    # Display the total number of leaf nodes.
    print("Number of Leaf Nodes :", model.get_n_leaves())

    print("\nFeature Importances:")

    # Display the importance of each input feature.
    for Feature, Importance in zip(X.columns, model.feature_importances_):

        print(Feature, ":", Importance)

    print("\n" + Border)


######################################################################
# Main Function
######################################################################

def main():

    # Call the Decision Tree Regression function.
    CaliforniaHousingDecisionTreeRegression( "dataset/california_housing.csv" )


######################################################################
# Starting Point of the Program
######################################################################

if __name__ == "__main__":
    main()