import streamlit as st
import pandas as pd
from utils.data_processing import validate_data

st.title("Modelo de Binomial Negativa")

uploaded_file = st.file_uploader("Carregue sua planilha (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if validate_data(data):
        st.success("Planilha carregada com sucesso!")
        st.dataframe(data.head())
    else:
        st.error("Erro: A planilha não segue o formato esperado.")
