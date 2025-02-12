import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('at.pkl1', 'rb') as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title("Logistic Regression Model Deployment")

# Description
st.write("""
### Predict using Logistic Regression Model
Enter the following details to get the prediction.
""")

# Define input fields for the features
def user_input_features():
    PassengerId = st.number_input('Passenger ID', min_value=1, step=1, value=1)  # Changed to numeric input
    Pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
    Name = st.text_input('Name (Enter any number)', value='0')  # User should enter numeric value to match training
    Sex = st.selectbox('Sex', ['male', 'female'])
    Age = st.number_input('Age', min_value=0, max_value=100, value=25)
    SibSp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
    Parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
    Ticket = st.text_input('Ticket Number (Enter any number)', value='0')  # User should enter numeric value
    Fare = st.number_input('Passenger Fare', min_value=0.0, value=32.0)
    Cabin = st.text_input('Cabin (Enter any number)', value='0')  # User should enter numeric value
    Embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
    
    # Convert categorical variables
    Sex = 1 if Sex == 'male' else 0
    Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]
    
    # Convert Name, Ticket, Cabin to numeric if possible
    try:
        Name = int(Name)
        Ticket = int(Ticket)
        Cabin = int(Cabin)
    except ValueError:
        st.error("Please enter numeric values for Name, Ticket Number, and Cabin.")

    data = {
        'PassengerId': PassengerId,
        'Pclass': Pclass,
        'Name': Name,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Ticket': Ticket,
        'Fare': Fare,
        'Cabin': Cabin,
        'Embarked': Embarked
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
data = user_input_features()

# Display user input
st.subheader('User Input:')
st.write(data)

# Make predictions
try:
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)

    # Display prediction result
    st.subheader('Prediction:')
    st.write('Survived' if prediction[0] == 1 else 'Did Not Survive')

    # Display prediction probability
    st.subheader('Prediction Probability:')
    st.write(prediction_proba)

except ValueError as e:
    st.error(f"Error: {e}")
