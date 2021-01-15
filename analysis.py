import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


def BI():
    st.subheader('Dataset')
    df = pd.read_csv('data/diabetes_data_upload.csv')  # default
    df1 = pd.read_csv('data/diabetes_data_upload_clean.csv')  # clean
    df2 = pd.read_csv('data/freqdist_of_age_data.csv')  # age
    st.dataframe(df)
    menu = st.sidebar.selectbox('', ['Information', 'Visualization'])
    if menu == 'Information':
        c1, c2 = st.beta_columns([2, 1])
        with c1:
            with st.beta_expander('Simple Analysis'):
                st.dataframe(df1.describe().transpose())

            with st.beta_expander('All Parameter'):
                st.write(df.columns)

        with c2:
            with st.beta_expander('Data Types'):
                st.dataframe(df.dtypes)
            with st.beta_expander('Patients'):
                st.dataframe(df['class'].value_counts())
                st.write({
                    'Positive': 'Diabetic Patient',
                    'Negetive': 'Non-Diabetic Patient'
                })

    else:
        st.subheader('Visualization')
        c1, c2 = st.beta_columns([2, 1])
        with c1:
            with st.beta_expander('Age outlier'):
                fig = px.box(df, x='Age', color=df['Gender'])
                st.plotly_chart(fig)

            with st.beta_expander('Correlation'):
                fig = plt.figure(figsize=(12, 8))
                sns.heatmap(df1.corr(), annot=True)
                st.pyplot(fig)

                st.subheader('Real Time correlation')
                fig1 = px.imshow(df1.corr())
                st.plotly_chart(fig1)

        with c2:
            with st.beta_expander('Gender Distribution'):
                fig = px.pie(names=df['Gender'], title='Male vs. Female', hole=0.4)
                fig.update_traces(textfont_size=15, textinfo='percent+label', textposition='inside')
                st.plotly_chart(fig, use_container_width=True)

            with st.beta_expander('Age Distribution'):
                st.dataframe(df2)
                fig = px.bar(data_frame=df2, x='Age', y='count', color=df2['Age'], text=df2['count'])
                st.plotly_chart(fig, use_container_width=True)

