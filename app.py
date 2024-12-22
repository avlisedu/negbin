import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices  # Para lidar com fórmulas em ZINB
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from utils.data_processing import validate_data

# Caminho para a logo
logo_path = "static/images/logo-favicon.svg"  # Atualize o caminho se necessário

# Exibindo informações da pesquisa e a logo
st.title("Modelo Zero-Inflated Negative Binomial (ZINB) com Painel")

st.write("""
    ### Informações da Pesquisa
    **Universidade Federal de Pernambuco**  
    **Departamento de Engenharia de Produção - DEP**  
    **Programa de Pós-Graduação em Engenharia de Produção - PPGEP**  
    **Dissertação de Mestrado**  
    **Discente**: Eduardo da Silva  
    **Orientadora**: Profa. Maísa Mendonça Silva  
""")

st.image(logo_path, use_container_width=True)

st.subheader("Upload de Planilha Padrão")
st.write("Por favor, insira uma planilha no formato padrão:")
st.write("- A célula A1 deve conter os nomes das variáveis.")
st.write("- A célula A2 deve indicar se cada variável é um fator (Sim/Não).")
st.write("- A partir da linha 3, a coluna A deve conter o índice (ano, mês e código IBGE) e as demais colunas os dados.")

# Botão para download do modelo padrão
modelo_path = "data/modelo_atualizado_com_fatores.xlsx"
with open(modelo_path, "rb") as file:
    st.download_button(
        label="Baixar Modelo Padrão Atualizado com Fatores (.xlsx)",
        data=file,
        file_name="modelo_atualizado_com_fatores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded_file = st.file_uploader("Carregue sua planilha em formato CSV ou XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Verificar o tipo de arquivo e carregar os dados
        if uploaded_file.name.endswith(".csv"):
            # Ler os nomes das variáveis (linha 1) e os fatores (linha 2)
            headers = pd.read_csv(uploaded_file, header=None, nrows=2)
            variable_names = headers.iloc[0, 1:].tolist()  # Nomes das variáveis
            factors_row = headers.iloc[1, 1:].tolist()  # Indicação de fatores

            # Ler os dados a partir da linha 3
            data = pd.read_csv(uploaded_file, skiprows=2, header=None)
        elif uploaded_file.name.endswith(".xlsx"):
            headers = pd.read_excel(uploaded_file, header=None, nrows=2)
            variable_names = headers.iloc[0, 1:].tolist()
            factors_row = headers.iloc[1, 1:].tolist()

            data = pd.read_excel(uploaded_file, skiprows=2, header=None)
        else:
            st.error("Formato de arquivo não suportado.")
            data = None

        if data is not None:
            # Ajustar os nomes das colunas
            data.columns = ["Índice"] + variable_names

            # Converter a coluna "Índice" para string
            data["Índice"] = data["Índice"].astype(str)

            # Processar o índice para separação correta de ano, mês e código IBGE
            data['Ano'] = data['Índice'].str[:4]  # Primeiros 4 caracteres são o ano
            data['Mês'] = data['Índice'].str[4:6]  # Próximos 2 caracteres são o mês
            data['Código IBGE'] = data['Índice'].str[6:]  # Restante é o código IBGE
            data.drop(columns=['Índice'], inplace=True)  # Remover a coluna original "Índice"

            # Exibição dos primeiros registros
            st.write("Pré-visualização dos dados carregados:")
            st.dataframe(data.head())

            # Validação do formato da planilha
            if validate_data(data):
                st.success("Planilha carregada e validada com sucesso!")

                # Seleção das colunas
                dependent_var = data.columns[0]  # Primeira coluna após ano, mês e código IBGE como variável dependente
                independent_vars = data.columns[1:-3]  # Demais colunas como variáveis explicativas

                st.write(f"Variável dependente: {dependent_var}")
                st.write(f"Variáveis explicativas: {', '.join(independent_vars)}")

                # Identificar fatores
                st.write("Configuração dos Fatores:")
                factors_mapping = dict(zip(variable_names, factors_row))
                for var, is_factor in factors_mapping.items():
                    st.write(f"- **{var}** é fator: {is_factor}")

                # Estatística descritiva
                st.subheader("Estatísticas Descritivas")
                st.write(data.describe())

                # Plotando gráficos da variável dependente
                st.subheader("Gráficos da Variável Dependente")
                fig, ax = plt.subplots()
                ax.hist(data[dependent_var], bins=10, color='blue', edgecolor='black')
                ax.set_title("Distribuição da Variável Dependente")
                ax.set_xlabel(dependent_var)
                ax.set_ylabel("Frequência")
                st.pyplot(fig)

                # Botão para iniciar a regressão
                if st.button("Iniciar Regressão ZINB"):
                    st.subheader("Regressão Zero-Inflated Negative Binomial (ZINB)")

                    # Corrigindo os nomes das variáveis para a fórmula
                    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"

                    # Separando a parte inflacionária e principal
                    y, X = dmatrices(formula, data, return_type='dataframe')

                    # Ajustando o modelo ZINB
                    model = ZeroInflatedNegativeBinomialP(y, X, inflation="logit")
                    results = model.fit()

                    # Formatando resultados
                    params = results.params
                    pvalues = results.pvalues
                    conf_int = results.conf_int()
                    significance = ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in pvalues]

                    # Criar DataFrame formatado
                    results_df = pd.DataFrame({
                        "Estimado": params,
                        "Erro Padrão": results.bse,
                        "p-valor": pvalues,
                        "Significância": significance
                    })
                    results_df.index.name = "Variável"

                    # Dividir em modelos Condicional e Inflacionário
                    cond_model = results_df.iloc[:len(X.columns)]
                    inflate_model = results_df.iloc[len(X.columns):]

                    # Exibir os modelos separados
                    st.write("### Modelo Condicional")
                    st.table(cond_model)

                    st.write("### Modelo de Inflação Zero")
                    st.table(inflate_model)

                    # Exibir métricas do modelo
                    st.write("### Métricas do Modelo")
                    st.write(f"- **AIC**: {results.aic:.2f}")
                    st.write(f"- **BIC**: {results.bic:.2f}")
                    st.write(f"- **Log-Likelihood**: {results.llf:.2f}")

            else:
                st.error("Erro: A planilha não segue o formato esperado. Certifique-se de que ela segue o padrão definido.")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
