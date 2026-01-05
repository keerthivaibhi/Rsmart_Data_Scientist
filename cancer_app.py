import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names


df = pd.DataFrame(X, columns=feature_names)
df['target'] = y


st.title("Breast Cancer Prediction App")
st.write("Enter patient details to predict if the tumor is benign or malignant")


st.sidebar.header("Input Features")
user_input = []
for feature in feature_names:
 
    value = st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
    user_input.append(value)


user_input = np.array(user_input).reshape(1, -1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
user_input_scaled = scaler.transform(user_input)


model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

st.subheader("Prediction")
st.write(f"The tumor is **{target_names[prediction[0]]}**")

st.subheader("Prediction Probability")
st.write(f"Benign: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Malignant: {prediction_proba[0][1]*100:.2f}%")


y_pred = model.predict(X_test)
st.subheader("Model Accuracy")
st.write(f"{accuracy_score(y_test, y_pred)*100:.2f}%")
