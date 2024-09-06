import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from patrick5day_rfmodel_068 import train_rf_model, plot_feature_importance
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

def plot_feature_influence(features, importances):
    plt.figure(figsize=(30, 18))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()  # Invert the y-axis for better readability
    st.pyplot(plt)

# Dados de importâncias (exemplo fornecido)
feature_importances = {
    'material_Cotton,Mesh,Polyester': 0.25,
    'height': 0.16,
    'thickness': 0.12,
    'width': 0.11,
    'brand_GroVia': 0.1,
    'brand_Thirsties': 0.1,
    'brand_bumGenius': 0.06,
    'material_Cotton': 0.06,
    'brand_FuzziBunz': 0.02,
    'material_Lead Free': 0.02,
    'material_Polyester': 0.01
}

# Convertendo o dicionário em DataFrame para visualização
feature_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])

# Configurando o layout do Streamlit
st.title('Análise de Características de Fraldas')

st.write("""
    Este painel permite analisar como diferentes características influenciam a avaliação dos clientes das fraldas.
    As características com maior importância são mais influentes nas avaliações.
""")

# Exibindo a tabela de importâncias
st.write('**Importâncias das Características:**')
st.dataframe(feature_df)

# Plotando a importância das características
st.write('**Gráfico de Importância das Características:**')
plot_feature_influence(feature_df['Feature'], feature_df['Importance'])

st.title('Random Forest Feature Importance')

# Train the model and get feature importance
rf_model, X_train = train_rf_model()

# Plot the feature importance using Plotly
fig = plot_feature_importance(rf_model, X_train)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
