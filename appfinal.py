import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Smart Fitness Tracker", layout="wide")

# ----------------------------
# TITLE & SIDEBAR
# ----------------------------
st.title("Smart Personal Fitness Tracker")
st.markdown("Helping you stay fit with insights & predictions")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1995/1995486.png", width=100)
st.sidebar.header("User Input Parameters")
st.sidebar.markdown("Provide your data below:")

# ----------------------------
# USER INPUT
# ----------------------------
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.5)
    duration = st.sidebar.slider("Duration (minutes)", 1, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temp (°C)", 36.0, 42.0, 37.5)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))
    gender_val = 1 if gender == "Male" else 0

    user_df = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender_val]
    })
    return user_df

input_df = user_input_features()

# ----------------------------
# LOAD DATA
# ----------------------------
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Combine datasets
exercise_df = pd.merge(exercise, calories, on="User_ID")
exercise_df.drop(columns=["User_ID"], inplace=True)
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

# Prepare data
df_model = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop("Calories", axis=1)
y = df_model["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Align input with training columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Predict
prediction = model.predict(input_df)[0]

# ----------------------------
# TABS FOR OUTPUT
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Insights", "About"])

with tab1:
    st.subheader("Calories Burned Prediction")
    st.success(f"You are likely to burn **{round(prediction, 2)} kilocalories**")
    st.write("Based on the information you provided:")
    st.dataframe(input_df, use_container_width=True)

with tab2:
    st.subheader("How Do You Compare?")

    # Comparison charts
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    sns.histplot(exercise_df['Calories'], bins=30, ax=axs[0,0], color="skyblue")
    axs[0,0].axvline(prediction, color='red', linestyle='--')
    axs[0,0].set_title("Calories Burned Distribution")

    sns.boxplot(x=exercise_df['Heart_Rate'], ax=axs[0,1], color="lightgreen")
    axs[0,1].axvline(input_df['Heart_Rate'].values[0], color='red', linestyle='--')
    axs[0,1].set_title("Heart Rate Comparison")

    sns.boxplot(x=exercise_df['Body_Temp'], ax=axs[1,0], color="orange")
    axs[1,0].axvline(input_df['Body_Temp'].values[0], color='red', linestyle='--')
    axs[1,0].set_title("Body Temp Comparison")

    sns.boxplot(x=exercise_df['Duration'], ax=axs[1,1], color="violet")
    axs[1,1].axvline(input_df['Duration'].values[0], color='red', linestyle='--')
    axs[1,1].set_title("Exercise Duration Comparison")

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Similar Users")
    similar = exercise_df[(exercise_df['Calories'] > prediction - 10) & (exercise_df['Calories'] < prediction + 10)]
    st.write(similar.sample(min(5, len(similar))))

with tab3:
    st.subheader(" About This App")
    st.write("""
        This Smart Fitness Tracker predicts how many kilocalories you burn during exercise
        based on your physiological data and workout duration.

        **Features:**
        - Real-time prediction with Random Forest
        - Visual comparisons with peer group
        - Personalized stats and similar case samples

        Created using [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).
    """)

# ----------------------------
# WATERMARK FOOTER
# ----------------------------
st.markdown("""
    <hr style='border-top: 1px solid #bbb;'>
    <div style='text-align: center; font-size: 14px;'>
        Developed by <b>Kartik Bansal</b> 
    </div>
""", unsafe_allow_html=True)
