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

  df = pd.read_csv('data_water_7k.csv')
  df_relevant = df[['Brand', 'Material', 'Color', 'Special Feature', 'Volume', 'rating_cat']]
  
  # Variáveis independentes e dependente
  X = df_relevant[['Brand', 'Material', 'Color', 'Special Feature', 'Volume']]
  y = df_relevant['rating_cat']
  
  # Inicializar o OneHotEncoder
  ohe = OneHotEncoder(sparse_output=False, drop='first')
  
  # Ajustar e transformar os dados
  X_encoded = ohe.fit_transform(X[['Brand', 'Material', 'Color', 'Special Feature']].astype(str))
  
  # Criar um DataFrame com as características codificadas
  X_encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(['Brand', 'Material', 'Color', 'Special Feature']))
  
  # Concatenar o DataFrame original (excluindo as colunas categóricas originais) com o novo DataFrame codificado
  X = pd.concat([X.drop(columns=['Brand', 'Material', 'Color', 'Special Feature']), X_encoded_df], axis=1)
  
  # Certificar que 'Volume' está no tipo númerico
  X['Volume'] = pd.to_numeric(X['Volume'], errors='coerce')
  
  # Dividir os dados em conjuntos de treino e teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  # Criar e treinar o modelo de floresta aleatória
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)
  
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
  
  # Helper function to get top 3 and sum "other" features
  def get_top_3_and_other(prefix, data):
      category_features = {k: v for k, v in data.items() if k.startswith(prefix)}
      sorted_features = sorted(category_features.items(), key=lambda x: x[1], reverse=True)
      
      top_3 = dict(sorted_features[:3])
      other = sum(v for _, v in sorted_features[3:])
      
      return top_3, other
  
  # Get top 3 and 'Other' for each category
  top_3_brand, other_brand = get_top_3_and_other('Brand', importances)
  top_3_material, other_material = get_top_3_and_other('Material', importances)
  top_3_color, other_color = get_top_3_and_other('Color', importances)
  top_3_special_feature, other_special_feature = get_top_3_and_other('Special Feature', importances)
  
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
      x=[importances['Volume']],
      orientation='h',
      marker=dict(color='lightgrey'),
      showlegend=True
  ))
  
  # Configure the layout
  fig.update_layout(
      title='Summed Feature Importances by Category (Top 3 Divided + Other)',
      xaxis_title='Importance',
      yaxis_title='Category',
      barmode='stack',
      template='plotly_white',
      legend_title_text='Top 3 Features + Other'
  )

  return fig
