import numpy as np
import streamlit as st
import os

from mlProject.pipeline.prediction import PredictionPipeline
# Setting the page title and icon
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered",
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [ "Predict", "Train"])

# Home page
# if page == "Home":
#     st.title("🍷 Wine Quality Prediction")
#     st.subheader("A Wine Quality Checking Web App")
    # st.image("static/assets/img/wine.jpg", use_column_width=True)

# Prediction page
if page == "Predict":
    st.title("Wine Quality Prediction")
    st.subheader("Please Fill in the Information")

    # Form for user input
    with st.form("prediction_form"):
        st.write("### Input Features:")
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
        citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
        chlorides = st.number_input("Chlorides", min_value=0.0, step=0.0001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, step=1)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, step=1)
        density = st.number_input("Density", min_value=0.0, step=0.0001)
        pH = st.number_input("pH", min_value=0.0, step=0.01)
        sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
        alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)
        data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
        data = np.array(data).reshape(1, 11)
        obj = PredictionPipeline()
        prediction = obj.predict(data)
        # Submit button
        submit = st.form_submit_button("Predict")

    # Display prediction result
    if submit:
        # Mock prediction logic (replace with actual model prediction)
        st.success(f"The predicted wine quality is: {prediction}")


# Train page
if page == "Train":
    st.title("Train the Model")
    st.subheader("Click the button below to train the model.")
    if st.button("Train"):
        os.system("dvc repro")
        st.success("Training Successful!")