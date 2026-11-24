import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


######################################################################
# Function Name : CaliforniaHousingLinearRegression
# Description   : Predict house prices using Simple Linear Regression
# Author        : Pranav Avinash Narkhede
######################################################################

def CaliforniaHousingLinearRegression(DataPath):

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

    # Select MedInc as the single input feature
    # Simple Linear Regression uses only one input feature
    X = df[["MedInc"]]

    # Y contains the target column that we want to predict
    Y = df["target"]

    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    print("Input Feature : MedInc")
    print("Target Variable : target")

    ##################################################################
    # Step 3 : Split the dataset into training and testing data
    ##################################################################

    print("\n", Border)
    print("Step 3 : Train-Test Split")
    print(Border)

    # Divide the dataset into 80% training and 20% testing data
    # random_state ensures the same split every time

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Training and testing data created successfully")

    print("X_train Shape : ", X_train.shape)
    print("X_test Shape  : ", X_test.shape)

    print("Y_train Shape : ", Y_train.shape)
    print("Y_test Shape  : ", Y_test.shape)

    ##################################################################
    # Step 4 : Create the Linear Regression model
    ##################################################################

    print("\n", Border)
    print("Step 4 : Create Linear Regression Model")
    print(Border)

    # Create a Simple Linear Regression model
    model = LinearRegression()

    print("Linear Regression model created successfully")

    ##################################################################
    # Step 5 : Train the model
    ##################################################################

    print("\n", Border)
    print("Step 5 : Model Training")
    print(Border)

    # Train the model using the training data
    model.fit(X_train, Y_train)

    print("Model training completed successfully")

    ##################################################################
    # Step 6 : Test the model
    ##################################################################

    print("\n", Border)
    print("Step 6 : Model Testing")
    print(Border)

    # Predict house prices using testing data
    Y_pred = model.predict(X_test)

    print("Model testing completed successfully")

    ##################################################################
    # Step 7 : Evaluate the model
    ##################################################################

    print("\n", Border)
    print("Step 7 : Model Evaluation")
    print(Border)

    # Calculate Mean Absolute Error
    MAE = mean_absolute_error(Y_test, Y_pred)

    # Calculate Mean Squared Error
    MSE = mean_squared_error(Y_test, Y_pred)

    # Calculate Root Mean Squared Error
    RMSE = np.sqrt(MSE)

    # Calculate R2 Score
    R2 = r2_score(Y_test, Y_pred)

    print("Mean Absolute Error (MAE) : ", MAE)
    print("Mean Squared Error (MSE) : ", MSE)
    print("Root Mean Squared Error (RMSE) : ", RMSE)
    print("R2 Score : ", R2)

    ##################################################################
    # Step 8 : Display model coefficients
    ##################################################################

    print("\n", Border)
    print("Step 8 : Model Coefficients")
    print(Border)

    # Display the coefficient and intercept of the model
    print("Coefficient : ", model.coef_[0])
    print("Intercept : ", model.intercept_)


######################################################################
# Function Name : main
# Description   : Starting point of the program
######################################################################

def main():

    # Start California Housing prediction using Simple Linear Regression
    CaliforniaHousingLinearRegression("dataset/california_housing.csv")


if __name__ == "__main__":
    main()