import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score 

Datapath = "student_performance_ml.csv"

Border = "-"*80

######################################################################################
#   Step 1 : Dataset loading
######################################################################################

print(Border)
print("Dataset loading")
print(Border)

df = pd.read_csv(Datapath)

print("Dataset loaded successfully")


######################################################################################
#   Step 2 : Data Analysis
######################################################################################

print(Border)
print("Data Analysis")
print(Border)

print("Null Values : \n",df.isnull().sum())

print("Shape of dataset : ",df.shape)

print("Columns names : ",df.columns)

print("Final Result Distribution:")
print(df["FinalResult"].value_counts())

print("Statistical report of dataset : ")
print(df.describe())

######################################################################################
#   Step 3 : Visualisation
######################################################################################

print(Border)
print("Visualisation")
print(Border)

plt.hist(df["StudyHours"] , bins= 10 ,  edgecolor = "black")

plt.title("Distribution of Student Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("No of Students")

plt.grid(axis="y" , alpha=0.3)

plt.show()

######################################################################################
#   Step 4 : Train test split
######################################################################################

print(Border)
print("Train test split")
print(Border)

features = ["StudyHours","Attendance" ,"PreviousScore","AssignmentsCompleted","SleepHours"]
labels = ["FinalResult"]
# deciding independent and dependent variables

X = df[features]
Y = df[labels]

# spliting the dataset for training and testing 

X_train ,X_test ,Y_train ,Y_test = train_test_split(X , Y , test_size=0.2 , random_state = 42)

print("X_train : ",X_train.shape)  
print("X_test : ",X_test.shape)     

print("Y_train : ",Y_train.shape)   
print("Y_test : ",Y_test.shape)

######################################################################################
#   Step 5 : Model Training
######################################################################################

print(Border)
print("Model Training")
print(Border)

# create the model object 

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train , Y_train)

######################################################################################
#   Step 6 : Model Testing
######################################################################################

print(Border)
print("Model Testing")
print(Border)

Y_pred = model.predict(X_test)

######################################################################################
#   Step 7 : Calculate Accuracy
######################################################################################

Y_train_pred = model.predict(X_train)
accuracy = accuracy_score(Y_train , Y_train_pred)

print("Training accuracy : ",accuracy*100)

accuracy = accuracy_score(Y_test , Y_pred)
print("Testing accuracy : ",accuracy*100)

