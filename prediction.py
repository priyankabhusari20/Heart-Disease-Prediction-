
# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
                                                                           
                                                                         
# loading and reading the dataset

heart = pd.read_csv("C:\\Users\\hp\Desktop\\flask3\\heart_cleaved_disease.csv")

# creating a copy of dataset so that will not affect our original dataset.
heart_df = heart.copy()

# Renaming some of the columns 
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head())

# model building 

#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

# creating K-Nearest-Neighbor classifier
model=RandomForestClassifier(n_estimators=20)
model.fit(x_train_scaler, y_train)
y_pred= model.predict(x_test_scaler)
p = model.score(x_test_scaler,y_test)
print(p)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Define a new DataFrame with the input data
new_data = pd.DataFrame({
    'age': [50],
    'sex': [1],
    'cp': [1],
    'trestbps': [150],
    'chol': [200],
    'fbs': [1],
    'restecg': [1],
    'thalach': [140],
    'exang': [0],
    'oldpeak': [0.8],
    'slope': [2],
    'ca': [2],
    'thal': [0]
})


# Scale the new data using the pre-fitted scaler
new_data_scaled = scaler.transform(new_data)

# Predict the class using the pre-trained model
predicted_class = model.predict(new_data_scaled)

# Output the prediction
if predicted_class[0] == 0:
    print('The person does not have heart disease.')
else:
    print('The person has heart disease.')

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(model, open(filename, 'wb'))







