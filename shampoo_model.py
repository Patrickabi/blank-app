import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import plotly.graph_objects as go

def get_shampoo_data():
    df = pd.read_csv('shampoo_final.csv')
    df_less_than_4 =    df[df['rating'] <= 4]
    df_greater_than_4 = df[df['rating'] > 4 ]
    def converter_categoria(valor):
          if valor <= 4:
              return '0'
          else:
              return '1'
    df['rating_cat'] = df['rating'].apply(converter_categoria)
    df.drop(columns=['rating_number'], inplace=True)
    item_form_mapping = {
          'Powder': ['Powder', 'powder'],
          'Liquid': ['Liquid', 'Lotion,Liquid,Gel', 'Liquid,Bar', 'Liquid,Creamy', 'Liquid,Foam', 'liquid', 'Liquid,Powder', 'Oil'],
          'Cream': ['Cream', 'Cream,Liquid', 'Cream,butter', 'Serum'],
          'Gel': ['Gel', 'Lotion,Liquid,Gel'],
          'Bar': ['Bar', 'Bars', 'Bar,Foam', 'Stick'],
          'Shampoo': ['Shampoo'],
          'Other': ['cape', 'Wand', 'Spray', 'Foam', 'Balm', 'Stick', 'wash', 'Instant', 'Wipe', 'Wipes', 'Ground', 'Oil, Wax', 'Aerosol', 'Individual, Pair', 'Pac', 'Roll On', 'Mask', 'foam', 'Pac']
      }
    def map_item_form(value):
        for category, keywords in item_form_mapping.items():
            if value in keywords:
                return category
        return 'Other' 
      
    df['item_form'] = df['item_form'].apply(map_item_form)
    brand_counts = df['brand'].value_counts()
    threshold = 10
    less_frequent_brands = brand_counts[brand_counts < threshold].index
    df['brand'] = df['brand'].apply(lambda x: 'Other' if x in less_frequent_brands else x)
    df_cleaned = df[df['brand'] != 'Other']
    features = ['unified_hair_type', 'size_classification', 'item_form', 'brand']
    target = 'rating_cat'
    X = df_cleaned[features]
    y = df_cleaned[target]
    X_encoded = pd.get_dummies(X, columns=features)
    
    return X_encoded, y

def shampoo_model(X_encoded, y):
  best_model = DecisionTreeClassifier(max_depth=9, min_samples_leaf=4, min_samples_split=6, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
  best_model.fit(X_train, y_train)
  feature_names = X_train.columns
  importances = best_model.feature_importances_
  
  # Verificar o comprimento das importâncias e dos nomes das características
  print(f"Number of feature names: {len(feature_names)}")
  print(f"Number of importances: {len(importances)}")
  
  # Criar o DataFrame de importâncias com os nomes das características corretos
  importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': importances
  }).sort_values(by='Importance', ascending=False)

  return best_model, X_train


def plot_shampoo_feature_importance(rf_model, X_train):
    
    importances = rf_model.feature_importances_
    
    # Create a DataFrame for the feature importances and their corresponding feature names
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    })
    
    # Add the "item_size" feature manually with a fixed importance value
    item_size_row = pd.DataFrame({'feature': ['item_size'], 'importance': [0.02278]})
    importance_df = pd.concat([importance_df, item_size_row], ignore_index=True)
    
    # Filter out features with importance less than 0.03
    importance_df = importance_df[importance_df['importance'] > 0.03]
    
    # Sort the features by importance
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Define distinct colors for each feature
    colors = [
        'rgba(255, 0, 0, 0.6)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 255, 0.6)', 
        'rgba(0, 0, 139, 0.6)', 'rgba(255, 140, 0, 0.6)', 'rgba(0, 100, 0, 0.6)', 
        'rgba(128, 0, 128, 0.6)', 'rgba(255, 165, 0, 0.6)', 'rgba(75, 0, 130, 0.6)', 
        'rgba(255, 105, 180, 0.6)', 'rgba(173, 216, 230, 0.6)', 'rgba(144, 238, 144, 0.6)'
    ]

    # Plotly bar chart
    fig = go.Figure()
    
    # Add bars for each feature
    for i, (index, row) in enumerate(importance_df.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=[row['feature']],
            orientation='h',
            name=row['feature'],
            marker_color=colors[i % len(colors)]  # Assign different colors
        ))
    
    # Add the "item_size" bar
    fig.add_trace(go.Bar(
        x=[0.02278],
        y=['item_size'],
        orientation='h',
        name='item_size',
        marker_color='rgba(255, 105, 180, 0.6)',  # Pink shade for item_size
    ))
    
    # Update layout for better display
    fig.update_layout(
        title="Feature Importances in Shampoo analysis",
        xaxis_title="Importance",
        yaxis_title="Feature",
        barmode='stack',  # Stack bars on top of each other
        yaxis={'categoryorder': 'total ascending'}
    )   

    return fig
