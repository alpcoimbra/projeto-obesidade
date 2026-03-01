# Dashboard de Predição de Obesidade

Aplicação desenvolvida em Streamlit para análise exploratória e predição de obesidade utilizando Machine Learning.

## Tecnologias Utilizadas
- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly

##  Funcionalidades

###  Sistema Preditivo
- Entrada de dados do paciente
- Classificação automática
- Probabilidade estimada de obesidade

###  Painel Analítico
- KPIs estratégicos
- Distribuição de obesidade
- Fatores associados
- Perfil antropométrico
- Importância das variáveis do modelo

##  Modelo Utilizado
Random Forest Classifier treinado com pipeline de pré-processamento (98% de acuracia)

Accuracy: 98.58 %
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       228
           1       0.98      0.98      0.98       195

    accuracy                           0.99       423
   macro avg       0.99      0.99      0.99       423
weighted avg       0.99      0.99      0.99       423


##  Como executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py

## Estrutura do Projeto
├── app.py
├── modelo_obesidade.pkl
├── Obesity.csv
├── requirements.txt
└── EDA/

---

Projeto acadêmico com fins educacionais.