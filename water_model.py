import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import randint
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

def plot_water_feature():

  importances = {
      'Volume': 0.1,
      'Brand_CamelBak': 0.02, 'Material_Polyester, Nylon, Polypropylene': 0.02, 
      'Special Feature_Carrying Loop, Straw': 0.02, 'Special Feature_nan': 0.02,
      'Brand_Contigo': 0.01, 'Brand_EcoVessel': 0.01, 'Brand_Hydro Flask': 0.01, 
      'Brand_Nalgene': 0.01, 'Brand_SIGG': 0.01, 'Brand_Simple Modern': 0.01, 
      'Brand_Under Armour': 0.01, 'Material_Aluminum': 0.01, 'Material_Eastman Tritan Copolyester': 0.01, 
      'Material_Metal': 0.01, 'Material_Other': 0.01, 'Material_Plastic': 0.01, 
      'Material_Stainless Steel': 0.01, 'Material_Steel': 0.01, 'Material_nan': 0.01, 
      'Color_Blue': 0.01, 'Color_Brushed Stainless': 0.01, 'Color_Charcoal': 0.01, 
      'Color_Foliage': 0.01, 'Color_Gray': 0.01, 'Color_Green': 0.01, 
      'Color_Hatching Dinos': 0.01, 'Color_Iguanas': 0.01, 'Color_Sapphire': 0.01, 
      'Color_nan': 0.01, 'Special Feature_Bpa Free,Dishwasher Safe': 0.01, 
      'Special Feature_Bpa Free,Dishwasher Safe,Leak Proof,Narrow Mouth': 0.01, 
      'Special Feature_Dishwasher Safe, Cold 24 hour, Spillproof': 0.01, 
      'Special Feature_Dishwasher Safe,Leak Proof,Narrow Mouth': 0.01, 
      'Special Feature_Durable': 0.01, 'Special Feature_Leak Proof, Insulated': 0.01, 
      'Special Feature_Wide Mouth, Leak Proof': 0.01
  }
  
  # Calculate the total sum of importances
  total_importance = sum(importances.values())
  
  # Scale the data so that the total sums up to 1
  scaled_importances = {k: v / total_importance for k, v in importances.items()}
  
  # Helper function to get top 3 and sum "other" features
  def get_top_3_and_other(prefix, data):
      category_features = {k: v for k, v in data.items() if k.startswith(prefix)}
      sorted_features = sorted(category_features.items(), key=lambda x: x[1], reverse=True)
      
      top_3 = dict(sorted_features[:3])
      other = sum(v for _, v in sorted_features[3:])
      
      return top_3, other
  
  # Get top 3 and 'Other' for each category
  top_3_brand, other_brand = get_top_3_and_other('Brand', scaled_importances)
  top_3_material, other_material = get_top_3_and_other('Material', scaled_importances)
  top_3_color, other_color = get_top_3_and_other('Color', scaled_importances)
  top_3_special_feature, other_special_feature = get_top_3_and_other('Special Feature', scaled_importances)
  
  # Add 'Other' category to each list
  top_3_brand['Other'] = other_brand
  top_3_material['Other'] = other_material
  top_3_color['Other'] = other_color
  top_3_special_feature['Other'] = other_special_feature
  
  # Define colors for the plot
  colors_brand = ['#1f77b4', '#aec7e8', '#c6dbef', '#d9d9d9']
  colors_material = ['#ff7f0e', '#ffbb78', '#fdd0a2', '#f5f5f5']
  colors_color = ['#2ca02c', '#98df8a', '#c7e9c0', '#e5e5e5']
  colors_special_feature = ['#d62728', '#ff9896', '#f7b6d2', '#e0e0e0']
  
  # Create the plotly figure
  fig = go.Figure()
  
  # Add bars for 'Brand'
  for i, (feature, value) in enumerate(top_3_brand.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Brand'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_brand[i]),
          showlegend=True
      ))
  
  # Add bars for 'Material'
  for i, (feature, value) in enumerate(top_3_material.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Material'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_material[i]),
          showlegend=True
      ))
  
  # Add bars for 'Color'
  for i, (feature, value) in enumerate(top_3_color.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Color'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_color[i]),
          showlegend=True
      ))
  
  # Add bars for 'Special Feature'
  for i, (feature, value) in enumerate(top_3_special_feature.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Special Feature'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_special_feature[i]),
          showlegend=True
      ))
  
  # Add 'Volume' bar separately
  fig.add_trace(go.Bar(
      name='Volume',
      y=['Volume'],
      x=[scaled_importances['Volume']],
      orientation='h',
      marker=dict(color='lightgrey'),
      showlegend=True
  ))
  
  # Configure the layout
  fig.update_layout(
      title='Summed Feature Importances by Category (Scaled to Sum 1)',
      xaxis_title='Importance (Scaled)',
      yaxis_title='Category',
      barmode='stack',
      template='plotly_white',
      legend_title_text='Top 3 Features + Other'
  )

  return fig
