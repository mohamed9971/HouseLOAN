import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

os.chdir('/Users/mohamedatef/Downloads')

Model = joblib.load('DT_houseLOAN.h5')
columns = joblib.load('DT_houseLOAN_columns.h5')

def main() :
    st.write('Welcome to House Loan Prediction')
    
    gender = st.selectbox('Please Select Gender' , ('Male' , 'Female'))
    married = st.selectbox('Are you married?' , ('Yes' , 'No'))
    dependents = st.selectbox('Do you have dependents?' , (0,1,2,3))
    education = st.selectbox('Are you graduate?' , ('Graduate' , 'Not Graduate'))
    employed = st.selectbox('Are You Employed?' , ('Yes','No'))
    app_income = st.number_input('Enter Your income')
    app_income = np.log(app_income)
    coapp_income = st.number_input('Enter coapplicant income')
    coapp_income = np.cbrt(coapp_income)
    loan_amount = st.number_input('Enter Loan Amount')
    loan_amount = np.log(loan_amount)
    loan_term = st.number_input('Enter loan amount term')
    credit = st.selectbox('Choose Credit History' , (1.0,0.0))
    property = st.selectbox('Choose property area' , ('Urban' , 'Semiurban','Rural'))
    prediction = 'Prediction is not made yet, Click Predict make prediction.'
    
    input_data = [gender,married,dependents,education,employed,app_income,coapp_income,loan_amount,loan_term,credit,property]
    
    input_df = pd.DataFrame([input_data] , columns=columns)
    
    if st.button('Predict Loan Approval.'):
        model_predict = Model.predict(input_df)[0]
        if model_predict == 1 :
            prediction = 'Loan Approved'
        else:
            prediction  = 'Loan Rejected'
        
    st.success(prediction)
    
    
    
if __name__ == '__main__' :
    main()
    
    