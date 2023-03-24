import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler




class KNNRegressor:
    def _init_(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array👍
        
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            dist = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            idx = np.argsort(dist)[:self.k]
            y_pred.append(np.mean(self.y[idx]))
        return np.array(y_pred)





st.title("Sales of retail stores")
st.header("Predicting Sales")




pickle_in = open("knn.pkl","rb")
lasso=pickle.load(pickle_in)



if nav == "Prediction":
    st.header("Know your sales Price")
    crim = st.number_input("Enter crim",0.01,0.10,step=0.01)
    zn = st.number_input("Enter zn",1.0,100.00,step=1.00)
    indus = st.number_input("Enter indus",1.0,15.00,step=1.00)
    chas = st.number_input("Enter chas",1.0,5.00,step=1.00)
    nox = st.number_input("Enter nox",0.01,1.00,step=0.01)
    rm = st.number_input("Enter rm",1.0,10.00,step=1.00)
    age = st.number_input("Enter your age",5.0,100.00,step=5.00)
    dis = st.number_input("Enter dis",1.0,10.00,step=1.00)
    rad = st.number_input("Enter rad",1.0,50.00,step=5.00)
    tax = st.number_input("Enter tax",100.0,500.00,step=50.00)
    ptratio = st.number_input("Enter ptratio",1.0,30.00,step=1.00)
    black = st.number_input("Enter black",100.0,500.00,step=50.00)
    lstat = st.number_input("Enter lstat",1.0,20.00,step=1.00)

    val = [[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,black,lstat]]
    val = np.array(val)
    X_test = pd.DataFrame(val)

    # Scaling the dataset
    X_test = scaler.fit_transform(X_test)


    
    pred = lasso.predict(X_test)[0]
    if st.button("Predict"):
        st.success(f"Your predicted price is {round(pred)}")



if nav == "About us":
    st.header("About us")
    st.write("Data Scientist Trainee")
    if st.button("Submit"):
        st.success("Submitted")
