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




st.title('Amazon Customer Satisfaction Visualizer')

st.write("""
    This panel helps us visualize the different impact of each feature when buying different kinds of products.
""")

st.write (""" Please, choose one of our products """)

option = st.selectbox(
    "",
    ("Diapers", "Shampoo", "Deodorants", "Water Bottles"),
)

if option == "Diapers":
    # Exibindo a tabela de importâncias
    st.write('**Features importance:**')
    st.dataframe(feature_df)
    
    st.title('Random Forest Feature Importance')
    # Train the model and get feature importance
    rf_model, X_train = train_rf_model()
    
    # Plot the feature importance using Plotly
    fig = plot_feature_importance(rf_model, X_train)
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
else:
    None
