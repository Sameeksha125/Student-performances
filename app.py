import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Student_Performance.csv")

df = load_data()

st.title("📊 Student Performance Analysis & Prediction App")

# Show raw data
with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df)

# Data Preprocessing
df_clean = df.copy()
le = LabelEncoder()
df_clean['Extracurricular Activities'] = le.fit_transform(df_clean['Extracurricular Activities'])

# Feature-target split
X = df_clean.drop('Performance Index', axis=1)
y = df_clean['Performance Index']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Section 1: Data Visualization
st.subheader("📈 Data Analysis")

# Plot correlation heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df_clean.corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Bar chart of average performance by activity
st.write("### Average Performance by Activity")
activity_perf = df.groupby("Extracurricular Activities")["Performance Index"].mean()
st.bar_chart(activity_perf)

# Section 2: Performance Prediction
st.subheader("🎯 Predict Student Performance")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider("Hours Studied", 0, 10, 5)
        previous_score = st.slider("Previous Scores", 0, 100, 75)
        activity = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    with col2:
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        sample_papers = st.slider("Sample Question Papers Practiced", 0, 10, 3)

    submitted = st.form_submit_button("Predict Performance")

    if submitted:
        input_data = pd.DataFrame({
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_score],
            'Extracurricular Activities': [1 if activity == "Yes" else 0],
            'Sleep Hours': [sleep_hours],
            'Sample Question Papers Practiced': [sample_papers]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"🎓 Predicted Performance Index: *{prediction:.2f}*")

# Section 3: Model Evaluation
with st.expander("📉 Model Evaluation Metrics"):
    st.write(f"*Mean Absolute Error (MAE)*: {mae:.2f}")
    st.write(f"*R² Score*: {r2:.2f}")