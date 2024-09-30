import streamlit as st
import numpy as np
import pandas as pd
import time


st.title("Welcome to the Smoker Status Prediction App! 🎉")
st.image("smoker1.png",width=500)
st.markdown("""
### We're glad you're here! 🚀
This application is designed to help predict whether an individual is a smoker or a non-smoker based on their health parameters. Simply enter your details, and our intelligent model will analyze the data and provide you with an accurate prediction.

#### How to Use:
1. Input your health information in the fields below.
2. Click on the **'Predict Smoker Status'** button.
3. Watch the progress as we analyze your data and see the results instantly!

Enjoy exploring, and take a step towards understanding health better! 💡""")
df = pd.read_csv("train.csv")


menu=st.sidebar.radio("Menu",["Home","Smoker Status"])
if menu=="Home":
    if st.title("Case Study on Smoker Analysis"):
        st.write("Shape of Dataset",df.shape)

    st.image("smoker3.png",width=550)



    if st.header("Tabular Data of Smoker status using Bio Signals"):
        st.table(df.head(20))

    if st.header("Statstical summary of Data"):
        st.table(df.describe())

    if st.header("Correlation graph"):
        correlation_matrix=df.corr()
        st.table(correlation_matrix.iloc[:-1,-1].sort_values(ascending=False))



if menu=="Smoker Status":
    st.title("Smoker status")

    from sklearn.ensemble import RandomForestClassifier
    random_forest_model = RandomForestClassifier(
    n_estimators=300,       
    criterion='gini',      
    max_depth=10,         
    min_samples_split=3,    
    min_samples_leaf=4,     
    random_state=15,
    max_features='log2',
    class_weight='balanced')

    X = df[['hemoglobin', 'height(cm)', 'weight(kg)', 'triglyceride', 'Gtp']]
    y = df['smoking']
    random_forest_model.fit(X, y)

    hemoglobin = st.number_input("Hemoglobin level:", min_value=0.0, step=0.1)
    height = st.number_input("Height (cm):", min_value=0.0, step=0.1)
    weight = st.number_input("Weight (kg):", min_value=0.0, step=0.1)
    triglyceride = st.number_input("Triglyceride level:", min_value=0.0, step=0.1)
    gtp = st.number_input("Gtp level:", min_value=0.0, step=0.1)
    
    smoking_effect_image_path = "smoker4.png"
    

    if st.button("Predict Smoker Status"):
        user_input = pd.DataFrame([[hemoglobin, height, weight, triglyceride, gtp]],
                              columns=['hemoglobin', 'height(cm)', 'weight(kg)', 'triglyceride', 'Gtp'])
    
        loading_text = st.text("LOADING...")
        progress_bar = st.progress(0)
        st.image(smoking_effect_image_path, caption="Smoking Effects on Health", use_column_width=True)

    # Simulate the prediction process with a loading effect
        for percent_complete in range(101):
            time.sleep(0.05)  # Adjust this sleep time to control the speed of the progress bar
            progress_bar.progress(percent_complete)
    
        prediction = random_forest_model.predict(user_input)

        if prediction[0] >=0.5:
            st.success("The model predicts that the person is a smoker.")
            st.write("**Effects of Smoking:** Smoking can lead to a wide range of health issues, including increased risk of heart disease, stroke, lung disease, and various cancers. It can also contribute to high blood pressure, weakened immune system, and reduced lung function.")
        else:
            st.success("The model predicts that the person is a non-smoker.")
            st.write("**Benefits of Not Smoking:** Non-smokers have a lower risk of developing heart disease, stroke, lung disease, and various cancers. They also generally have better lung function, stronger immunity, and overall improved health and well-being.")
    else:
        st.warning("Please enter valid input values for all fields.")