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
        time.sleep(0.005)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    
    st.header("The study", divider=True)
    st.write ("""In our review of the key factors influencing diaper purchase decisions, we conducted an in-depth analysis of customer preferences.
    By examining various features, such as brand, dimensions, material quality, and others, we identified which one of them plays the biggest rows in the final decision.
    Our findings revealed that :red[Brand], :blue[Volume], :violet[Material] and :green[Weight] stand out as the most influential factors.
    These elements play a pivotal role in shaping customer decisions, highlighting the significance of brand loyalty, product capacity, and material composition in the diaper selection process.""")

    st.image("cortado.png", caption="Customer decision tree")

    _diaper_text = """
    As we can see, the first choice the customer should make is regarding the type of diaper. Disposable diapers surely are more practical, but cloth diapers can be easy on the budget.
    As for the disposable ones, a small percentege of customers do worry about the "Lead-free" label, ensuring that none of the heavy metal used in some inks are present in the composition.
    We can even see the importance of good brand names showing, as "Mama Bear" products act as a remarkable divisor for good reviews
    """


    def stream_data():
        for word in _diaper_text.split(" "):
            yield word + " "
            time.sleep(0.02)


    if st.button("but why ?"):
        st.write_stream(stream_data)

    st.title('Visualizing')

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
