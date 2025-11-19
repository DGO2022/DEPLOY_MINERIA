'''
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. Cargar modelo
MODEL_PATH = Path("artefactos") / "modelo_pima.pkl"
modelo = joblib.load(MODEL_PATH)

st.title("🤖 Predicción de Diabetes (Pima Dataset)")

# 2. Ingreso de datos del paciente
st.write("Ingrese los valores clínicos para predecir si la paciente probablemente tiene diabetes.")
data = {
    'npreg': st.slider("Número de embarazos", 0, 20, 2),
    'glu':   st.slider("Nivel de glucosa (mg/dl)", 50, 200, 100),
    'bp':    st.slider("Presión arterial (mmHg)", 40, 130, 70),
    'skin':  st.slider("Espesor del pliegue cutáneo (mm)", 7, 100, 20),
    'bmi':   st.slider("IMC", 10.0, 50.0, 25.0),
    'ped':   st.slider("Pedigree de diabetes", 0.0, 2.5, 0.5),
    'age':   st.slider("Edad (años)", 18, 90, 35)
}

# 3. Predicción
if st.button("Predecir"):
    entrada = pd.DataFrame([data])
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]
    resultado = "Diabética" if pred == 1 else "No diabética"
    st.write(f"Resultado: **{resultado}**")
    st.write(f"Probabilidad estimada: **{prob:.2f}**")




# --- agregamos mas

# Coeficientes del modelo
import pandas as pd
coef_df = pd.DataFrame({
    'Variable': modelo.feature_names_in_,
    'Peso': modelo.coef_[0]
}).sort_values(by='Peso', key=abs, ascending=False)

st.subheader("Importancia de cada variable en la predicción")
st.bar_chart(coef_df.set_index("Variable"))

# EDA

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

df = sm.datasets.get_rdataset("Pima.tr", "MASS").data
df['type'] = df['type'].map({'Yes': 1, 'No': 0})

st.subheader("Correlación entre variables")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
'''