import streamlit as st
import pickle
import pandas as pd

# 1. Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. App Title and Description
st.title("🎓 Student Marks Prediction App")
st.write("""
This app predicts a student's final **Marks** based on their daily habits and academic background. 
Please adjust the inputs below to see the predicted score.
""")

# 3. Create Input Fields for User
st.header("Enter Student Details:")

study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85, step=1)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
previous_score = st.number_input("Previous Exam Score", min_value=0, max_value=100, value=75, step=1)
internet_hours = st.number_input("Internet Usage Hours per Day", min_value=0.0, max_value=24.0, value=2.0, step=0.5)

# 4. Prediction Button and Output
if st.button("Predict Marks"):
    # Organize inputs into a DataFrame to preserve feature names matching the training dataset
    input_data = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Attendance_Percentage': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Score': previous_score,
        'Internet_Hours': internet_hours
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Cap the results logically between 0 and 100
    final_prediction = max(0.0, min(100.0, prediction))
    
    # Display Result
    st.success(f"### Predicted Marks: {final_prediction:.2f} / 100")