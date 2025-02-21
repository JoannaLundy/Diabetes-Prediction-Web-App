# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generic entry point script."""
import os
import streamlit as st
import numpy as np
import tensorflow as tf

import os
import streamlit as st

import numpy as np
import tensorflow as tf
import requests
from tensorflow.keras.models import load_model

# Load the trained model
model_path = load_model("diabetes_model.keras")


st.title("Diabetes Prediction Web App")
st.write("Enter your health details below to predict the risk of diabetes.")

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction Button
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, diabetes_pedigree, age]])
if st.button("Predict Diabetes Risk"):
    prediction = model_path.predict(user_input)
    result = "Diabetic💉" if prediction[0][0] > 0.5 else "Non-Diabetic✨"
    st.write(f"**Prediction: {result}**")


