import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

st.set_page_config(page_title="Prediction of Disease Outbreaks", page_icon=":rocket:")
diabetes_model = pickle.load(open('model/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('model/heart_model.sav', 'rb'))
parkinson_model = pickle.load(open('model/parkinsons_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreaks', ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Disease Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", 0, 17, 3)
        skin_thickness = st.number_input("Skin Thickness value", 0, 99, 23)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.0, 2.4, 0.3725)
    with col2:
        glucose = st.number_input("Glucose level", 0, 199, 117)
        insulin = st.number_input("Insulin level", 0, 846, 30)
        age = st.number_input("Age", 21, 81, 29)
    with col3:
        blood_pressure = st.number_input("Blood Pressure value", 0, 122, 72)
        bmi = st.number_input("BMI value", 0.0, 67.1, 32.0)
    
    if st.button("Predict Diabetes"):
        diabetes_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        diabetes_prediction = diabetes_model.predict(diabetes_features)
        
        if diabetes_prediction[0] == 1:
            st.success("The person have diabetes.")
        else:
            st.success("The person not have diabetes.")

if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 0, 100, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])  # 0: Female, 1: Male
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 0, 200, 120)
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])  # 0: False, 1: True
        restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", 0, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])  # 0: No, 1: Yes
    
    with col3:
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
        chol = st.number_input("Serum Cholestoral in mg/dl", 0, 600, 200)
    
    if st.button("Predict Heart Disease"):
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0
        
        heart_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_prediction = heart_model.predict(heart_features)
        
        if heart_prediction[0] == 1:
            st.success("The person is likely to have heart disease.")
        else:
            st.success("The person is not likely to have heart disease.")

if selected == "Parkinson Disease Prediction":
    st.title("Parkinson Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 0.0, 300.0, 119.992)
        fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 600.0, 157.302)
        flo = st.number_input("MDVP:Flo(Hz)", 0.0, 300.0, 74.997)
        jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.00784)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.1, 0.00007)
        rap = st.number_input("MDVP:RAP", 0.0, 1.0, 0.00370)
        spread1 = st.number_input("Spread1", -10.0, 0.0, -4.813031)
        ppe = st.number_input("PPE", 0.0, 1.0, 0.284654)
    
    with col2:
        ppq = st.number_input("MDVP:PPQ", 0.0, 1.0, 0.00554)
        ddp = st.number_input("Jitter:DDP", 0.0, 1.0, 0.01109)
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.04374)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 1.0, 0.426)
        apq3 = st.number_input("Shimmer:APQ3", 0.0, 1.0, 0.02182)
        apq5 = st.number_input("Shimmer:APQ5", 0.0, 1.0, 0.03130)
        spread2 = st.number_input("Spread2", 0.0, 1.0, 0.266482)
    
    with col3:
        apq = st.number_input("MDVP:APQ", 0.0, 1.0, 0.02971)
        dda = st.number_input("Shimmer:DDA", 0.0, 1.0, 0.06545)
        nhr = st.number_input("NHR", 0.0, 1.0, 0.02211)
        hnr = st.number_input("HNR", 0.0, 50.0, 21.033)
        rpde = st.number_input("RPDE", 0.0, 1.0, 0.414783)
        dfa = st.number_input("DFA", 0.0, 1.0, 0.815285)
        d2 = st.number_input("D2", 0.0, 5.0, 2.301442)
        
    
    if st.button("Predict Parkinson's Disease"):
        parkinsons_features = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        parkinsons_prediction = parkinson_model.predict(parkinsons_features)
        
        if parkinsons_prediction[0] == 1:
            st.success("The person is likely to have Parkinson's disease.")
        else:
            st.success("The person is not likely to have Parkinson's disease.")