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

    rf_model, X_train = train_rf_model()
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 2, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    
    
    st.write ("""In our review of the key factors influencing diaper purchase decisions, we conducted an in-depth analysis of customer preferences.
    By examining various features, such as brand, product volume, material quality, and others, we identified the major driving forces behind consumer choices.
    Our findings revealed that Brand, Volume, and Material stand out as the most influential factors.
    These elements play a pivotal role in shaping customer decisions, highlighting the significance of brand loyalty, product capacity, and material composition in the diaper selection process.""")

    

    st.title('Random Forest Feature Importance')

    # Plot the feature importance 
    fig = plot_feature_importance(rf_model, X_train)

    on = st.toggle("Show Graph")

    if on:
        st.plotly_chart(fig)

elif option == "Shampoo":
    with st.spinner('Wait for it...'):
        time.sleep(5)
        st.success("Done!")

elif option == "Deodorants":
    None
