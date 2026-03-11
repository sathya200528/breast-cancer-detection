import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

st.title("🔬 Breast Cancer Detection App")
st.write("Move the sliders and click Predict!")

st.subheader("Enter Patient Details:")

radius = st.slider("Mean Radius", 6.0, 30.0, 14.0)
texture = st.slider("Mean Texture", 9.0, 40.0, 19.0)
perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 92.0)
area = st.slider("Mean Area", 140.0, 2500.0, 654.0)
smoothness = st.slider("Mean Smoothness", 0.05, 0.16, 0.10)

input_data = X.mean().values.copy()
input_data[0] = radius
input_data[1] = texture
input_data[2] = perimeter
input_data[3] = area
input_data[4] = smoothness

input_df = pd.DataFrame([input_data], columns=data.feature_names)

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Result: BENIGN (Not Cancer)")
    else:
        st.error("❌ Result: MALIGNANT (Cancer Detected)")