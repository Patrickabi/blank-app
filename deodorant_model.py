import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import plotly.express as px
import plotly.graph_objects as go

def get_deodo_data():
  df = pd.read_csv('deodorants_final.csv')
  
  # Define rating_cat as customer satisfaction (1 for yes, 0 for no)
  def converter_categoria(valor):
      if valor <= 4:
          return '0'
      else:
          return '1'
  df['rating_cat'] = df['rating'].apply(converter_categoria)
  df.drop(columns=['rating_number'], inplace=True)
  cols_to_drop = ['main_category', 'average_rating',
                  'price', 'images_x', 'videos', 'store',
                  'bought_together', 'images_y',
                  'user_id', 'helpful_vote', 'verified_purchase']
  
  df = df.drop(columns=cols_to_drop)
  brand_counts = df['Brand'].value_counts()
  top_50_brands = brand_counts[:50]
  df_filtered = df[df['Brand'].isin(top_50_brands.index)]
  scent_counts = df_filtered['Scent'].value_counts()
  top_50_scent = scent_counts[:50]
  df_filtered_scent = df_filtered[df_filtered['Scent'].isin(top_50_scent.index)]

  def standardize_item_form(item_form):
      if pd.isna(item_form):
          return 'Unknown'
      item_form = item_form.lower()
      if 'spray' in item_form:
          return 'Spray'
      elif 'stick' in item_form:
          return 'Stick'
      elif 'powder' in item_form:
          return 'Powder'
      elif 'roll on' in item_form or 'roll-on' in item_form:
          return 'Roll On'
      elif 'gel' in item_form:
          return 'Gel'
      elif 'cream' in item_form:
          return 'Cream'
      elif 'liquid' in item_form:
          return 'Liquid'
      elif 'balm' in item_form:
          return 'Balm'
      elif 'wipe' in item_form:
          return 'Wipe'
      else:
          return 'Other'
  df_filtered_scent['Item Form'] = df_filtered_scent['Item Form'].apply(standardize_item_form)
  item_form_counts = df_filtered_scent['Item Form'].value_counts()
  top_item_forms = item_form_counts[item_form_counts >= 20].index
  df_filtered_scent = df_filtered_scent[df_filtered_scent['Item Form'].isin(top_item_forms)]
  Item_Form_counts = df_filtered_scent['Item Form'].value_counts()
  Item_Form_counts.head(50)
  df_final = df_filtered_scent
  df_encoded = df_final.copy()
  encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' remove uma categoria para evitar multicolinearidade
  encoded_columns = encoder.fit_transform(df[['Item Form', 'Brand', 'Scent']])
  encoded_column_names = encoder.get_feature_names_out(['Item Form', 'Brand', 'Scent'])

  df_encoded = pd.DataFrame(encoded_columns, columns=encoded_column_names)

  df_final = pd.concat([df.drop(columns=['Item Form', 'Brand', 'Scent']), df_encoded], axis=1)

  df_final.columns[:50]
  X = df_final.drop(['title_x','features','description', 'categories', 'details',
       'parent_asin', 'rating', 'title_y', 'text', 'asin','timestamp', 'rating_cat'], axis=1)
  y = df_final['rating_cat']
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  best_params = {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 4}

  return df_final, X_train, y_train

def deodo_model(X_train, y_train):
    
  best_params = {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 4}
  clf = DecisionTreeClassifier(**best_params, random_state=42)
  # Treinando o modelo com os melhores parâmetros
  clf.fit(X_train, y_train)
  return clf

def plot_deodo_feature_importance(rf_model, X_train):
    
  importances = clf.feature_importances_
  
  # Create a DataFrame for the feature importances and their corresponding feature names
  importance_df = pd.DataFrame({
      'feature': X_train.columns,
      'importance': importances
  })
  
  # Filter out features with importance less than 0.02
  importance_df = importance_df[importance_df['importance'] > 0.02]
  
  # Separate features related to "Brand", "Item form", and "Scent"
  brand_df = importance_df[importance_df['feature'].str.contains('Brand', case=False, regex=True)]
  scent_df = importance_df[importance_df['feature'].str.contains('Scent', case=False, regex=True)]
  item_form_df = importance_df[importance_df['feature'].str.contains('Item form', case=False, regex=True)]
  
  # Group by "Brand" and "Scent", summing their importances
  brand_importance_sum = brand_df['importance'].sum()
  scent_importance_sum = scent_df['importance'].sum()
  
  # Plotly bar chart with stacked bars
  fig = go.Figure()
  
  # Add the "Item form" bar as a standalone bar
  fig.add_trace(go.Bar(
      x=[item_form_df['importance'].sum()],
      y=['Item form'],
      orientation='h',
      name='Item form',
      marker_color='rgba(0, 255, 0, 0.6)'  # Green for Item form
  ))
  
  # Add "Brand" as a stacked bar with its sub-features inside
  for i, (index, row) in enumerate(brand_df.iterrows()):
      fig.add_trace(go.Bar(
          x=[row['importance']],
          y=['Brand (summed)'],
          orientation='h',
          name=row['feature'],
          marker_color=f'rgba(255, {100 + i * 30}, 0, 0.6)',  # Gradient of orange for each brand feature
          showlegend=True  # Show legend for individual brands
      ))
  
  # Add "Scent" as a stacked bar with its sub-features inside
  for i, (index, row) in enumerate(scent_df.iterrows()):
      fig.add_trace(go.Bar(
          x=[row['importance']],
          y=['Scent (summed)'],
          orientation='h',
          name=row['feature'],
          marker_color=f'rgba(0, {100 + i * 30}, 255, 0.6)',  # Gradient of blue for each scent feature
          showlegend=True  # Show legend for individual scents
      ))
  
  # Update layout for better display
  fig.update_layout(
      title="Feature Importances in Random Forest (Summed and Stacked for Brand and Scent)",
      xaxis_title="Importance",
      yaxis_title="Feature",
      barmode='stack',  # Stack bars on top of each other
      yaxis={'categoryorder': 'total ascending'}
  )


  return fig
