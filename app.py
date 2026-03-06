import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e features
modelo = joblib.load('modelo_risco.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title='Previsão de Risco Escolar', layout='centered')

st.title('Previsão de Risco de Defasagem')
st.write('Preencha os indicadores do aluno para estimar a probabilidade de risco.')

# Entradas do usuário
IAA = st.number_input('IAA - Autoavaliação', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
IEG = st.number_input('IEG - Engajamento', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
IPS = st.number_input('IPS - Psicossocial', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
IPP = st.number_input('IPP - Psicopedagógico', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
IDA = st.number_input('IDA - Aprendizagem', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
IPV = st.number_input('IPV - Ponto de Virada', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
IAN = st.number_input('IAN - Adequação ao Nível', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
INDE = st.number_input('INDE - Nota Global', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
Defasagem = st.number_input('Defasagem', min_value=0.0, max_value=10.0, value=0.0, step=1.0)

if st.button('Calcular risco'):
    entrada = pd.DataFrame([{
        'IAA': IAA,
        'IEG': IEG,
        'IPS': IPS,
        'IPP': IPP,
        'IDA': IDA,
        'IPV': IPV,
        'IAN': IAN,
        'INDE': INDE,
        'Defasagem': Defasagem
    }])

    prob_risco = modelo.predict_proba(entrada)[0][1]
    classe = modelo.predict(entrada)[0]

    st.subheader('Resultado')
    st.write(f'**Probabilidade de risco:** {prob_risco:.2%}')

    if prob_risco < 0.30:
        st.success('Baixo risco')
    elif prob_risco < 0.70:
        st.warning('Risco moderado')
    else:
        st.error('Alto risco')

    st.write(f'**Classificação do modelo:** {classe}')