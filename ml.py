import joblib
import os
import pandas as pd
import numpy as np
import streamlit as st

gender_encoded = {'Male': 1, 'Female': 0}
label_encoded = {'Yes': 1, 'No': 0}
target_encoded = {'Positive': 1, 'Negative': 0}


@st.cache
def loadModel():
    loaded_model = joblib.load(open('models/logistic_regression_model_diabetes_21_oct_2020.pkl', 'rb'))
    return loaded_model


def gender_mapping(gender, gender_dict):
    for key, val in gender_dict.items():
        if key == gender:
            return val


def feature_mapping(value):
    feature_encoded = {'Yes': 1, 'No': 0}
    for key, val in feature_encoded.items():
        if key == value:
            return val


def ML():
    st.subheader('Dataset')
    df = pd.read_csv('data/diabetes_data_upload.csv')  # default
    df1 = pd.read_csv('data/diabetes_data_upload_clean.csv')  # clean
    df2 = pd.read_csv('data/freqdist_of_age_data.csv')  # age

    st.dataframe(df)
    age = st.sidebar.number_input('Age', 10, 100)
    gen = st.sidebar.radio('Gender', ['Male', 'Female'])

    with st.sidebar.beta_expander('A. Symptoms'):
        py = st.radio('Polyuria', ['Yes', 'No'])
        pa = st.radio('Polydipsia', ['Yes', 'No'])
        swl = st.radio('Sudden Weight Loss', ['Yes', 'No'])
        w = st.radio('Weakness', ['Yes', 'No'])
        pp = st.radio('Polyphagia', ['Yes', 'No'])
        gt = st.radio('Genital Thrush', ['Yes', 'No'])

    with st.sidebar.beta_expander('B. Symptoms'):
        vb = st.radio('Visual Blurring', ['Yes', 'No'])
        it = st.radio('Itching', ['Yes', 'No'])
        ir = st.radio('Irritability', ['Yes', 'No'])
        dh = st.radio('Delayed Healing', ['Yes', 'No'])
        par = st.radio('Partial Paresis', ['Yes', 'No'])
        ppa = st.radio('Muscle Stiffness', ['Yes', 'No'])
        ap = st.radio('alopecia', ['Yes', 'No'])
        ob = st.radio('Obesity', ['Yes', 'No'])

    result = {
        'Age': age,
        'Gender': gen,
        'Polyuria': py,
        'Polydipsia': pa,
        'Sudden Weight Loss': swl,
        'Weakness': w,
        'Polyphagia': ppa,
        'Genital Thrush': gt,
        'Visual Blurring': vb,
        'Itching': it,
        'Irritability': ir,
        'Delayed Healing': dh,
        'Partial Paresis': par,
        'Muscle Stiffness': ppa,
        'alopecia': ap,
        'Obesity': ob,
    }
    encoded_result = []
    
    c1, c2, c3 = st.beta_columns([2, 1, 1])
    with c1:
        with st.beta_expander('Your Selected options'):
            st.write(result)

    with c2:
        with st.beta_expander('Encoded Options'):
            for i in result.values():
                if type(i) == int:
                    encoded_result.append(i)
                elif i in ['Male', 'Female']:
                    gender = gender_mapping(gen, gender_encoded)
                    encoded_result.append(gender)
                else:
                    encoded_result.append(feature_mapping(i))

            st.write(encoded_result)

    with c3:
        with st.beta_expander('Result'):
            sample = np.array(encoded_result).reshape(1, -1)
            model = loadModel()
            ml_result = model.predict(sample)
            if ml_result[[0]] == 1:
                st.write({
                    'Diabetic': 'Yes'
                })
            else:
                st.write({
                    'Diabetic': 'No'
                })
