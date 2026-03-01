import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

@st.cache_resource
def load_model():
    return joblib.load("modelo_obesidade.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Obesity.csv")

# Carregar modelo
model = load_model()

st.set_page_config(page_title="Predição de Obesidade", layout="centered")

tab1, tab2 = st.tabs(["Predição", "Insights Analíticos"])

# ======================
# ABA 1
# ======================

with tab1:
    st.title("Sistema Preditivo de Obesidade")
    st.markdown("Preencha as informações do paciente para estimar risco de obesidade.")

    # =====================
    # ENTRADAS DO USUÁRIO
    # =====================
    age = st.number_input("Idade", min_value=14, max_value=100, value=30)
    height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)

    gender = st.selectbox("Gênero", ["Male", "Female"])
    family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
    favc = st.selectbox("Consome alimentos calóricos frequentemente?", ["yes", "no"])

    fcvc = st.slider("Frequência de consumo de vegetais (1–3)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais (1–4)", 1, 4, 3)
    ch2o = st.slider("Consumo de água diário (1–3)", 1, 3, 2)
    faf = st.slider("Frequência de atividade física (0–3)", 0, 3, 1)
    tue = st.slider("Tempo de tela (0–2)", 0, 2, 1)

    caec = st.selectbox("Consome lanches entre refeições?", 
                        ["no", "Sometimes", "Frequently", "Always"])

    smoke = st.selectbox("Fuma?", ["yes", "no"])
    scc = st.selectbox("Monitora calorias?", ["yes", "no"])

    calc = st.selectbox("Consumo de álcool", 
                        ["no", "Sometimes", "Frequently", "Always"])

    mtrans = st.selectbox("Meio de transporte", 
                        ["Automobile", "Motorbike", "Bike", 
                        "Public_Transportation", "Walking"])

    # =====================
    # BOTÃO DE PREVISÃO
    # =====================
    if st.button("Realizar Previsão"):
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history': [family_history],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [caec],
            'SMOKE': [smoke],
            'CH2O': [ch2o],
            'SCC': [scc],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc],
            'MTRANS': [mtrans]
        })

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Paciente classificado como OBESO")
        else:
            st.success("Paciente classificado como NÃO OBESO")

        st.write(f"Probabilidade estimada de obesidade: {proba:.2%}")

# ======================
# ABA 2
# ======================

with tab2:
    st.title("Painel Analítico - Insights sobre Obesidade")

    # =========================
    # CARREGAMENTO DOS DADOS
    # =========================
    df = load_data()

    obese_classes = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    df['Obeso_bin'] = df['Obesity'].apply(
        lambda x: 1 if x in obese_classes else 0
    )

    df['IMC'] = df['Weight'] / (df['Height'] ** 2)

    # =========================
    # FILTROS INTERATIVOS
    # =========================
    st.sidebar.header("Filtros do Painel")

    genero = st.sidebar.selectbox(
        "Gênero",
        options=["Todos"] + sorted(df['Gender'].unique())
    )

    idade_min, idade_max = st.sidebar.slider(
        "Faixa Etária",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (18, 60)
    )

    historico = st.sidebar.selectbox(
        "Histórico Familiar",
        options=["Todos"] + sorted(df['family_history'].unique())
    )

    favc_filter = st.sidebar.selectbox(
        "Consumo frequente de alimentos calóricos (FAVC)",
        options=["Todos"] + sorted(df['FAVC'].unique())
    )

    faf_min, faf_max = st.sidebar.slider(
        "Frequência de Atividade Física (FAF)",
        float(df['FAF'].min()),
        float(df['FAF'].max()),
        (float(df['FAF'].min()), float(df['FAF'].max()))
    )

    # Aplicação dos filtros
    if genero != "Todos":
        df = df[df['Gender'] == genero]

    if historico != "Todos":
        df = df[df['family_history'] == historico]

    df = df[(df['Age'] >= idade_min) & (df['Age'] <= idade_max)]

    if favc_filter != "Todos":
        df = df[df['FAVC'] == favc_filter]

    df = df[(df['FAF'] >= faf_min) & (df['FAF'] <= faf_max)]

    # =========================
    # BLOCO 1 - KPIs (BIG NUMBERS)
    # =========================
    st.markdown("## Indicadores Estratégicos")

    total = len(df)
    perc_obesos = round(df['Obeso_bin'].mean() * 100, 1) if total > 0 else 0
    media_idade = round(df['Age'].mean(), 1) if total > 0 else 0
    media_imc = round(df['IMC'].mean(), 1) if total > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Pacientes", total)
    k2.metric("% Obesos", f"{perc_obesos}%")
    k3.metric("Idade Média", media_idade)
    k4.metric("IMC Médio", media_imc)

    st.markdown("---")

    # =========================
    # BLOCO 2 - DISTRIBUIÇÃO GERAL
    # =========================
    st.markdown("## Distribuição da Obesidade")

    col1, col2 = st.columns([2, 1])

    with col1:
        if not df.empty:
            dist = df['Obeso_bin'].value_counts().reset_index()
            dist.columns = ['Obesidade', 'Quantidade']
            dist['Obesidade'] = dist['Obesidade'].map({0: 'Não Obeso', 1: 'Obeso'})

            fig = px.bar(
                dist,
                x='Obesidade',
                y='Quantidade',
                text='Quantidade',
                title='Distribuição de Obesidade'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Nenhum dado disponível para os filtros selecionados.")

    with col2:
        st.markdown(f"""
        **Insight Executivo:**
        - {perc_obesos}% da população filtrada apresenta obesidade.
        - A prevalência varia conforme filtros aplicados.
        """)

    st.markdown("---")

    # =========================
    # BLOCO 3 - FATORES ASSOCIADOS
    # =========================
    st.markdown("## Fatores Associados")

    col3, col4 = st.columns(2)

    with col3:
        if df.empty:
            st.warning("Sem dados para os filtros selecionados.")
        else:
            cross = pd.crosstab(
                df['family_history'],
                df['Obeso_bin'],
                normalize='index'
            ) * 100

            cross = cross.reset_index().melt(
                id_vars='family_history',
                value_name='Percentual',
                var_name='Obeso_bin'
            )

            cross['Obeso_bin'] = cross['Obeso_bin'].map({0: 'Não Obeso', 1: 'Obeso'})

            fig2 = px.bar(
                cross,
                x='family_history',
                y='Percentual',
                color='Obeso_bin',
                barmode='group',
                title="Percentual por Histórico Familiar"
            )
            st.plotly_chart(fig2, width="stretch")

    with col4:
        if not df.empty:
            fig3 = px.box(
                df,
                x='Obeso_bin',
                y='FAF',
                points=False,
                title="Atividade Física por Grupo"
            )
            fig3.update_xaxes(tickvals=[0, 1], ticktext=['Não Obeso', 'Obeso'])
            st.plotly_chart(fig3, width="stretch")

    st.markdown("---")

    # =========================
    # BLOCO 4 - ANÁLISE ANTROPOMÉTRICA
    # =========================
    st.markdown("## Perfil Antropométrico")

    if not df.empty:
        fig4 = px.box(
            df,
            x='Obeso_bin',
            y='IMC',
            points=False,
            title="Distribuição de IMC por Grupo"
        )
        fig4.update_xaxes(tickvals=[0, 1], ticktext=['Não Obeso', 'Obeso'])
        st.plotly_chart(fig4, width="stretch")

    st.markdown("""
    **Interpretação Clínica:**
    Observa-se clara separação de IMC entre os grupos, reforçando coerência epidemiológica.
    """)

    st.markdown("---")

    # =========================
# BLOCO 5 - IDADE VS OBESIDADE (CATEGORIAS)
# =========================
    st.markdown("## Idade vs Categoria de Obesidade")

    if not df.empty:

        fig5 = px.box(
            df,
            x='Obesity',   # coluna categórica (Normal_Weight, Obesity_Type_I, etc)
            y='Age',
            title="Idade vs Obesidade",
            points="outliers"
        )

        fig5.update_layout(
            xaxis_title="Categoria de Obesidade",
            yaxis_title="Idade",
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig5, width="stretch")

    else:
        st.warning("Sem dados para os filtros selecionados.")