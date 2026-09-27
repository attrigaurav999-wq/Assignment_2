
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('churn_logistic_model.pkl')
scaler = joblib.load('churn_scaler.pkl')

# Get feature names from the original X dataframe (assuming X is available from previous cells)
# In a real deployed app, you'd load these from a predefined list or a dummy dataframe
# For this example, we'll manually list them, ensure they match the training order.
feature_names = ['Call  Failure', 'Complains', 'Subscription  Length', 'Charge  Amount',
                 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers',
                 'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value']

st.set_page_config(page_title='Customer Churn Prediction App', layout='centered')

st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict if they will churn.')

# Create input fields for each feature
with st.form('churn_prediction_form'):
    st.header('Customer Details')

    call_failure = st.number_input('Call Failure', min_value=0, value=0)
    complains = st.radio('Complains', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=1)
    subscription_length = st.slider('Subscription Length (months)', min_value=1, max_value=60, value=30)
    charge_amount = st.number_input('Charge Amount', min_value=0, value=0)
    seconds_of_use = st.number_input('Seconds of Use', min_value=0, value=1000)
    frequency_of_use = st.number_input('Frequency of Use', min_value=0, value=20)
    frequency_of_sms = st.number_input('Frequency of SMS', min_value=0, value=10)
    distinct_called_numbers = st.number_input('Distinct Called Numbers', min_value=0, value=5)
    age_group = st.slider('Age Group', min_value=1, max_value=5, value=3) # Assuming categories like 1: <20, 2: 20-30 etc.
    tariff_plan = st.radio('Tariff Plan', options=[1, 2], format_func=lambda x: f'Plan {x}', index=0)
    status = st.radio('Status', options=[1, 2], format_func=lambda x: f'Status {x}', index=0)
    age = st.slider('Age', min_value=18, max_value=90, value=30)
    customer_value = st.number_input('Customer Value', min_value=0.0, value=100.0, format='%.2f')

    submitted = st.form_submit_button('Predict Churn')

    if submitted:
        # Create a DataFrame from inputs
        input_data = pd.DataFrame([[call_failure, complains, subscription_length, charge_amount,
                                    seconds_of_use, frequency_of_use, frequency_of_sms, distinct_called_numbers,
                                    age_group, tariff_plan, status, age, customer_value]],
                                  columns=feature_names)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0][1] # Probability of churn

        st.subheader('Prediction Results')
        if prediction[0] == 1:
            st.error(f"The customer is predicted to CHURN with a probability of {prediction_proba:.2f}")
        else:
            st.success(f"The customer is predicted NOT to churn with a probability of {1 - prediction_proba:.2f}")

        st.write("--- Feature Inputs ---")
        st.write(input_data)
