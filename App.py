import numpy as np
import pickle
import pandas as pd
import streamlit as st 


class KNNRegressor:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            dist = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            idx = np.argsort(dist)[:self.k]
            y_pred.append(np.mean(self.y[idx]))
        return np.array(y_pred)


pickle_in = open("scaler_sclr.pkl","rb")
knn=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def Sales_Of_Retail_Store(week_id,outlet,product_identifier,department_identifier,category_of_product,state,day,month,year):
    
    """Let's Predict the sales of retail store 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: week_id
        in: query
        type: number
        required: true
      - name: outlet
        in: query
        type: number
        required: true
      - name: product_identifier
        in: query
        type: number
        required: true
      - name: department_identifier
        in: query
        type: number
        required: true
      - name: category_of_product
        in: query
        type: number
        required: true
      - name: state
        in: query
        type: number
        required: true
      - name: day
        in: query
        type: number
        required: true
      - name: month
        in: query
        type: number
        required: true
      - name: year
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
            
    """
    prediction=knn.predict([[week_id,outlet,product_identifier,department_identifier,category_of_product,state,day,month,year]])
    print(prediction)
    return prediction

def main():
    st.title("Sale of Retail Store")
    week_id = st.text_input("week_id","Type Here")
    outlet = st.text_input("outlet","Type Here")
    product_identifier = st.text_input("product_identifier","Type Here")
    department_identifier = st.text_input("department_identifier","Type Here")
    category_of_product = st.text_input("category_of_product","Type Here")
    state = st.text_input("state","Type Here")
    day = st.text_input("day","Type Here")
    month = st.text_input("month","Type Here")
    year = st.text_input("year","Type Here")  
  
    if st.button("Predict"):
        result=Sales_Of_Retail_Store([week_id,outlet,product_identifier,department_identifier,category_of_product,state,day,month,year])
    st.success('The output is {}'.format(result))


if __name__=='__main__':
    main()
