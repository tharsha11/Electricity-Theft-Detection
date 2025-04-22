import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
st.title('Machine Learning Based Electricity Theft Detection')
# Load the dataset
df = pd.read_csv("R:\projects\electricity theft detection\csp new\dataset.csv")
# Remove unwanted columns
df.drop(['date', 'id'], axis=1, inplace=True)
# Split the data into features (x) and target (y)
x = df.drop(['flag'], axis=1)
y = df['flag']
# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)
# Create a dictionary for model names and model instances
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='linear'),
    'DTree': DecisionTreeClassifier(),
    'RForest': RandomForestClassifier(n_estimators=100),
    'Logistic': LogisticRegression(),
    'NaiveBias': GaussianNB(),
    'KMeans': KMeans(n_clusters=2),
    'Gradientboster': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=100, max_features=5)
}
# Train and evaluate each model
model_accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy
# User input section
st.header("Predict Faithfulness of a Customer")
st.subheader("Enter customer features:")
feature_names = x.columns.tolist()
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:", min_value=0.0)
# Predict using Logistic Regression
lr = models['SVM']
user_input_features = [user_input[feature] for feature in feature_names]
user_input_scaled = scaler.transform([user_input_features])
prediction = lr.predict(user_input_scaled)
st.subheader("Prediction:")
if prediction == 0:
    st.write("Faithful")
else:
    st.write("Unfaithful")
model_names = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())
st.write("Model Accuracies:")
st.write(model_accuracies)
# Create a bar plot of model accuracies
st.header("Classifier Model Comparision")
st.bar_chart(accuracies)
