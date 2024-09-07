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
    df = create_df()

    df = df.fillna('Unknown')  # Or another appropriate value
    
    # Explicitly convert relevant columns to objects
    df['Brand'] = df['Brand'].astype('object')
    df['Material Type'] = df['Material Type'].astype('object')
    df['rating_cat'] = df['rating_cat'].astype(int)

    # Create ipyvizzu Object with the DataFrame
    obj = create_vizzu_obj(df)

    # Preset plot usage. Preset plots work directly with DataFrames.
    bar_obj = bar_chart(df,
                        x="Material Type", 
                        y="rating_cat",
                        title="1.Using preset plot function `bar_chart()`")

    # Animate with defined arguments 
    anim_obj = beta_vizzu_animate(bar_obj,
                                  x="Brand",
                                  y="rating_cat",
                                  title="Animate with beta_vizzu_animate () function",
                                  label="Brand",
                                  color="Brand",  # Ensure 'Size' exists
                                  legend="color",
                                  sort="byValue",
                                  reverse=True,
                                  align="center",
                                  split=False)

    # Animate with general dict-based arguments 
    _dict = {
        "size": {"set": "Brand"}, 
        "geometry": "circle",
        "coordSystem": "polar",
        "title": "Animate with vizzu_animate () function",
    }
    anim_obj2 = vizzu_animate(anim_obj, _dict)

    # Visualize within Streamlit
    if st.button("Animate"):
        vizzu_plot(anim_obj2)
