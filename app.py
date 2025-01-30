import streamlit as st
import pandas as pd
import plotly.express as px
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from patsy import dmatrices


#ATUAL
# Configuração inicial
st.set_page_config(
    page_title="Modelos com Dados em Painel",
    page_icon="static/images/gpsid.png",
    layout="wide"
)

# Logo e título
st.image("static/images/gpsid.png", use_container_width=False, width=200)
st.title("Modelos com Dados em Painel")

# Informações da pesquisa
st.markdown("""
### Informações da Pesquisa
- **Universidade Federal de Pernambuco**
- **Departamento de Engenharia de Produção - DEP**
- **Programa de Pós-Graduação em Engenharia de Produção - PPGEP**
- **Dissertação de Mestrado**
- **Discente**: Eduardo da Silva
- **Orientadora**: Profa. Maísa Mendonça Silva
""")
st.divider()

# Upload de dados
uploaded_file = st.file_uploader("Carregue sua planilha em formato CSV ou XLSX", type=["csv", "xlsx"])

data = None
if uploaded_file:
    try:
        # Carregar dados
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file, encoding="utf-8")
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado.")

        if data is not None:
            st.subheader("Pré-visualização dos Dados")
            st.dataframe(data.head())

            # **Tratamento de Valores Ausentes**
            st.markdown("### Tratamento de Valores Ausentes")
            remove_na = st.checkbox("Remover linhas com valores ausentes")
            if remove_na:
                data = data.dropna()
                st.write("Dados atualizados após remoção de valores ausentes:")
                st.dataframe(data.head())

            # **Cartões Informativos**
            st.markdown("### Resumo dos Dados")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total de Observações", len(data))
            with col2:
                st.metric("Número de Variáveis", len(data.columns))
            with col3:
                st.metric("Média da Variável Dependente", round(data.iloc[:, -1].mean(), 2))  # Última coluna como exemplo
            with col4:
                st.metric("Valores Ausentes", data.isnull().sum().sum())

            # **Seleção de Variáveis**
            st.markdown("### Seleção de Variáveis")
            col_time, col_dependent = st.columns(2)

            # Seleção do tempo e dependente
            with col_time:
                time_var = st.selectbox("Selecione a variável que representa o tempo:", data.columns)

            with col_dependent:
                dependent_var = st.selectbox("Selecione a variável dependente (alvo):", data.columns)

            # Seleção de variáveis explicativas
            selected_vars = st.multiselect(
                "Selecione as variáveis explicativas (X):",
                [col for col in data.columns if col not in [time_var, dependent_var]]
            )

            # Seleção de variáveis dicotômicas
            binary_vars = []
            if st.checkbox("Sua planilha contém variáveis binárias/dicotômicas (0 ou 1)?"):
                binary_vars = st.multiselect(
                    "Selecione as variáveis binárias/dicotômicas:",
                    selected_vars
                )

            # Exibir seleção final
            st.write("### Variáveis Selecionadas")
            st.write(f"Tempo: {time_var}")
            st.write(f"Dependente: {dependent_var}")
            st.write(f"Explicativas: {', '.join(selected_vars)}")
            st.write(f"Dicotômicas: {', '.join(binary_vars)}")

            # **Gráfico da Variável Dependente**
            st.markdown("### Distribuição da Variável Dependente")
            fig = px.histogram(
                data,
                x=dependent_var,
                nbins=10,
                title="Distribuição da Variável Dependente",
                labels={dependent_var: "Valores", "count": "Frequência"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig.update_layout(template="simple_white")
            st.plotly_chart(fig, use_container_width=True)

            # **Escolha de Modelo**
            st.markdown("### Escolha de Modelo")
            model_choice = st.radio(
                "Selecione o tipo de modelo:",
                ("Pooled", "Efeito Fixo", "Efeito Aleatório", "ZINB")
            )

            # Botão para executar o modelo
            if st.button("Executar Modelo"):
                try:
                    # Remover valores ausentes nas variáveis selecionadas
                    all_vars = [dependent_var] + selected_vars
                    model_data = data.dropna(subset=all_vars)

                    # Formulação da fórmula
                    formula = f"{dependent_var} ~ {' + '.join(selected_vars)}"

                    # Preparar os dados no formato correto
                    y, X = dmatrices(formula, model_data, return_type="dataframe")

                    if model_choice == "Pooled":
                        st.subheader("Resultados do Modelo: Pooled")
                        model = PooledOLS(y, X)
                        results = model.fit()
                        st.text(results.summary)

                    elif model_choice == "Efeito Fixo":
                        st.subheader("Resultados do Modelo: Efeito Fixo")
                        model = PanelOLS(y, X, entity_effects=True)
                        results = model.fit()
                        st.text(results.summary)

                    elif model_choice == "Efeito Aleatório":
                        st.subheader("Resultados do Modelo: Efeito Aleatório")
                        model = RandomEffects(y, X)
                        results = model.fit()
                        st.text(results.summary)

                    elif model_choice == "ZINB":
                        st.subheader("Resultados do Modelo: ZINB")
                        model = ZeroInflatedNegativeBinomialP(y, X, inflation="logit")
                        results = model.fit()
                        st.text(results.summary)

                except Exception as e:
                    st.error(f"Erro ao processar o modelo: {e}")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
