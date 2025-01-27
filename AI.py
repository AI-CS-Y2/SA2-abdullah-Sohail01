import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data_path = 'ev_charging_patterns.csv'
ev_data = pd.read_csv(data_path)

# Preprocessing: Drop rows with missing values in relevant columns
ev_data = ev_data.dropna(subset=['Charging Duration (hours)', 'Distance Driven (since last charge) (km)', 'State of Charge (End %)'])

# Create a binary target variable: `passed` (1 if SoC End > 80, else 0)
ev_data['passed'] = (ev_data['State of Charge (End %)'] > 80).astype(int)

# Define features and target
X = ev_data[['Charging Duration (hours)', 'Distance Driven (since last charge) (km)']]
y = ev_data['passed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict with KNN
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {knn_accuracy * 100:.2f}%')

# Initialize and train Logistic Regression classifier
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict with Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%')

# Loop for making predictions
while True:
    # Input commands for user to enter charging duration and distance driven
    charging_duration = float(input("Enter Charging Duration (hours): "))
    distance_driven = float(input("Enter Distance Driven (since last charge) (km): "))

    # Create a DataFrame for the new data
    new_data = pd.DataFrame([[charging_duration, distance_driven]], columns=['Charging Duration (hours)', 'Distance Driven (since last charge) (km)'])

    # Make predictions using KNN and Logistic Regression
    new_prediction_knn = knn.predict(new_data)
    new_prediction_log_reg = log_reg.predict(new_data)
    print(f'KNN Prediction for new data {new_data.values[0]}: {new_prediction_knn[0]}')
    print(f'Logistic Regression Prediction for new data {new_data.values[0]}: {new_prediction_log_reg[0]}')

    another = input("\nWould you like to make another prediction? (yes/no): ").lower()
    if another != 'yes':
        print("Thank you for using the EV Charging Predictor!")
        break