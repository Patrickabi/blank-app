import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from patrick5day_rfmodel_068 import train_rf_model, plot_feature_importance, create_df
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import time
from st_vizzu import *
import visgraph

st.set_page_config(layout="wide", page_title="Customer Decision Tree - Amazon")

st.title('Amazon Customer Satisfaction Visualizer')

st.write("""
    This panel helps us visualize the different impact of each feature when buying different kinds of products.
""")

st.write (""" Please, choose one of our products """)
option = None
option = st.selectbox(
    "",
    ("Select one", "Diapers", "Shampoo", "Deodorants", "Water Bottles"),
)


if option == "Diapers":


    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


    st.title('Random Forest Feature Importance')
    # Train the model and get feature importance
    rf_model, X_train = train_rf_model()
    
    # Plot the feature importance using Plotly
    fig = plot_feature_importance(rf_model, X_train)
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
elif option == "Shampoo":
    with st.spinner('Wait for it...'):
        time.sleep(5)
        st.success("Done!")

elif option == "Deodorants":
    None
