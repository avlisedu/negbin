import streamlit as st
import pandas as pd
import plotly.express as px
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from patsy import dmatrices

# oi ana
# Configuração inicial
st.set_page_config(
    page_title="Modelos com Dados em Painel",
    page_icon="static/images/reg.png",
    layout="wide"
)



logo_path = "static/images/gpsid.png"
st.image(logo_path, use_container_width=False, width=200)

# Título e descrição
st.title("📊 Modelos de Regressão com Dados em Painel")
st.markdown("""
## 🎓 Informações da Pesquisa
- **🏛 Universidade Federal de Pernambuco**
- **📌 Departamento de Engenharia de Produção - DEP**
- **📚 Programa de Pós-Graduação em Engenharia de Produção - PPGEP**
- **📖 Dissertação de Mestrado**
- **👨‍🎓 Discente**: Eduardo da Silva  
- **👩‍🏫 Orientadora**: Profa. Maísa Mendonça Silva  
""", unsafe_allow_html=True)

st.divider()

# Upload de dados
st.markdown("## 📂 Upload de Dados")
uploaded_file = st.file_uploader("📎 Carregue sua planilha (CSV ou XLSX)", type=["csv", "xlsx"])

data = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.markdown("### ⚙️ Opções de Leitura do CSV")
            col_sep_options = {"Vírgula (,)": ",", "Ponto e vírgula (;)": ";", "Tabulação (Tab)": "\t", "Espaço ( )": " "}
            selected_sep = st.selectbox("🛠 Selecione o separador:", list(col_sep_options.keys()))
            col_sep = col_sep_options[selected_sep]
            encoding = st.selectbox("🌐 Codificação do arquivo", ["utf-8", "latin1", "iso-8859-1"])
            data = pd.read_csv(uploaded_file, sep=col_sep, encoding=encoding)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Formato de arquivo não suportado.")

        if data is not None:
            st.divider()
            st.markdown("## 🔍 Pré-visualização dos Dados")
            preview_rows = st.slider("🔢 Selecione o número de linhas para visualizar", 5, 50, 10)
            st.dataframe(data.head(preview_rows))

            st.divider()
            st.markdown("## 🛠 Tratamento de Dados")
            remove_na = st.checkbox("🧹 Remover linhas com valores ausentes")
            if remove_na:
                data = data.dropna()
                st.write("✅ Dados atualizados após remoção de valores ausentes:")
                st.dataframe(data.head(preview_rows))

            st.divider()
            st.markdown("## 🎯 Seleção de Variáveis")
            col_indiv, col_time, col_dependent = st.columns(3)
            with col_indiv:
                indiv_var = st.selectbox("👤 Variável do indivíduo:", data.columns)
            with col_time:
                time_var = st.selectbox("⏳ Variável de tempo:", data.columns)
            with col_dependent:
                dependent_var = st.selectbox("🎯 Variável dependente (alvo):", data.columns)
            selected_vars = st.multiselect("📊 Variáveis explicativas (X):", [col for col in data.columns if col not in [time_var, dependent_var, indiv_var]])
            binary_vars = []
            if st.checkbox("⚖️ O conjunto de dados contém variáveis binárias (0 ou 1)?"):
                binary_vars = st.multiselect("📌 Selecione as variáveis binárias:", selected_vars)
            
            st.divider()
            st.markdown("## 📊 Escolha de Modelo")
            model_choice = st.radio("📌 Selecione o tipo de modelo:", ("Pooled", "Efeito Fixo", "Efeito Aleatório", "ZINB"))
            inflation_vars = []
            if model_choice == "ZINB":
                inflation_vars = st.multiselect("⚠️ Selecione as variáveis para modelagem de inflação (zero-inflated):", selected_vars)
            
            st.divider()
            if st.button("🚀 Executar Modelo"):
                try:
                    model_data = data.dropna(subset=[dependent_var] + selected_vars + inflation_vars)
                    model_data = model_data.set_index([indiv_var, time_var])
                    formula = f"{dependent_var} ~ {' + '.join(selected_vars)}"
                    y, X = dmatrices(formula, model_data, return_type="dataframe")
                    
                    if model_choice == "Pooled":
                        model = PooledOLS(y, X, check_rank=False)
                        results = model.fit()
                        st.metric("📌 R²", f"{results.rsquared:.4f}")
                        st.metric("📉 F-Estatística", f"{results.f_statistic.stat:.4f}")
                    elif model_choice == "Efeito Fixo":
                        model = PanelOLS(y, X, entity_effects=True)
                        results = model.fit()
                        st.metric("📌 R²", f"{results.rsquared:.4f}")
                        st.metric("📉 F-Estatística", f"{results.f_statistic.stat:.4f}")
                    elif model_choice == "Efeito Aleatório":
                        model = RandomEffects(y, X)
                        results = model.fit()
                        st.metric("📌 R²", f"{results.rsquared:.4f}")
                        st.metric("📉 F-Estatística", f"{results.f_statistic.stat:.4f}")
                    elif model_choice == "ZINB":
                        zinb_formula = f"{dependent_var} ~ {' + '.join(selected_vars)}"
                        zinb_inflation_formula = f"{' + '.join(inflation_vars)} ~ 1"
                        y_zinb, X_zinb = dmatrices(zinb_formula, model_data, return_type="dataframe")
                        _, X_infl = dmatrices(zinb_inflation_formula, model_data, return_type="dataframe")
                        model = ZeroInflatedNegativeBinomialP(endog=y_zinb, exog=X_zinb, exog_infl=X_infl, inflation="logit")
                        results = model.fit()
                        st.metric("📌 Log-Likelihood", f"{results.llf:.4f}")
                        st.metric("📉 AIC", f"{results.aic:.4f}")
                        st.metric("📈 BIC", f"{results.bic:.4f}")
                    
                    st.divider()
                    st.markdown("## 🔹 **Coeficientes do Modelo**")
                    if model_choice in ["Pooled", "Efeito Fixo", "Efeito Aleatório"]:
                        coef_table = pd.DataFrame({
                            "Coeficiente": results.params,
                            "Erro Padrão": results.std_errors,
                            "T-Valor": results.tstats
                        })
                    elif model_choice == "ZINB":
                        coef_table = pd.DataFrame({
                            "Coeficiente": results.params,
                            "Erro Padrão": results.bse,
                            "Z-Valor": results.tvalues,
                            "P-Valor": results.pvalues
                        })
                    st.dataframe(coef_table)
                except Exception as e:
                    st.error(f"❌ Erro ao processar o modelo: {e}")
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")
