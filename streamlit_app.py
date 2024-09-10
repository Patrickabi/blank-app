import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from diaper_model import train_rf_model, plot_feature_importance, create_df
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
from shampoo_model import *
from deodorant_model import get_deodo_data, deodo_model, plot_deodo_feature_importance
from water_model import *


sidebar_logo = "logo (1).png"
st.logo(sidebar_logo)

st.image("logo.png")

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
        time.sleep(0.0005)
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
        time.sleep(0.005)
        st.success("Done!")

    st.header("The study", divider=True)
    st.write ("""This chart illustrates the feature importances from our Random Forest model, where each bar represents the relative contribution of different product
    attributes to the consumer's purchasing decision. As observed, the :red[brand] of the product significantly influences consumer choice, accounting for over :rainbow[97% of the decision-making] process.
    This highlights the dominant role that brand reputation and recognition play in shaping consumer preferences in the shampoo market.""")

    st.image("shampoo.png", caption="Customer decision tree")
    
    on = st.toggle("Show Graph")
    X, y = get_shampoo_data()
    model, X_train = shampoo_model(X, y)
    fig = plot_shampoo_feature_importance(model, X_train)


    shampoo_text = """Other features, such as ingredient composition and item size, play a much smaller role. However,
    they still contribute to the overall decision, as consumers may also consider the product’s size and specific ingredients 
    when making their choice. Despite this, it is clear from our study that brand loyalty and recognition are the primary factors driving purchasing behavior, overshadowing other product attributes."""
    

    def stream_data_sh():
        for word in shampoo_text.split(" "):
            yield word + " "
            time.sleep(0.02)


    if st.button("but why ?"):
        st.write_stream(stream_data_sh)
    
    if on:
        st.plotly_chart(fig)

elif option == "Deodorants":
    time.sleep(0.005)
    st.success("Done!")
    
    deodo_text = """Our analysis revealed that among the features considered, the brand of the deodorant emerged as the most influential factor in the purchasing decision.
    This finding was consistent across various segments of the dataset. Consumers demonstrated a strong preference for well-known and trusted brands,
    indicating that brand recognition and reputation play a critical role in their purchase choices.
    In comparison, while item form (such as spray, stick, or roll-on) and scent were also important factors, they did not have as significant an impact on consumer decisions as the brand did.
    Item form and scent preferences varied among consumers but were secondary to brand influence.
    The research highlights the paramount importance of brand in consumer decision-making for deodorants. While other factors like item form and scent do affect consumer choices,
    brand loyalty and recognition are the dominant factors driving purchase decisions. This insight can be valuable for companies looking to position their products effectively in the market
    and for understanding consumer behavior trends in the personal care industry."""
    
    st.header("The study", divider=True)
    st.write("""We analyzed a comprehensive dataset of deodorant products, focusing on several key features: :red[brand],
    :blue[item form], and :green[scent]. These features were chosen due to their potential impact on consumer preferences.
    We utilized various data analysis techniques to determine the significance of each feature in the decision-making process.""")

    st.image("deodorant.png", caption="Customer decision tree")

    def stream_data_deo():
        for word in deodo_text.split(" "):
            yield word + " "
            time.sleep(0.02)
    if st.button("but why ?"):
        st.write_stream(stream_data_deo)
    
    df, x, y = get_deodo_data()
    deodo_model = deodo_model(x, y)
    fig = plot_deodo_feature_importance(deodo_model, x)
    
    on = st.toggle("Show Graph")
    if on:
        st.plotly_chart(fig)

elif option == "Water Bottles":
    time.sleep(0.005)
    st.success("Done!")

    st.header("The study", divider=True)
    st.write("""In this project, we conducted an in-depth analysis of customer decision-making when purchasing water bottles using a Random Forest model.
    The goal was to identify the key factors influencing purchase decisions based on historical data and product characteristics. Our researchers were able to identify that
    :blue[Volume], :red[Color], :violet[Material] and :red[Brand] play the biggest row when choosing a product""")

    water_text = """The analysis revealed that the most important factors for customers when choosing a water bottle were volume, color, material, and brand.
    Among these, volume stood out as the most significant factor, indicating that customers prioritize the bottle’s capacity in their purchasing decisions.
    This insight can help companies better tailor their product offerings to meet customer preferences and improve their marketing strategies."""

    fig = plot_water_feature()
    def stream_data_water():
        for word in water_text.split(" "):
            yield word + " "
            time.sleep(0.02)
    if st.button("but why ?"):
        st.write_stream(stream_data_water)     
    on = st.toggle("Show Graph")
    if on:
        st.plotly_chart(fig)
    
