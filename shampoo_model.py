import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

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
    
    # Filter for 'Brand' columns
    brand_features = importance_df[importance_df['feature'].str.contains('Brand')]
    
    # Find the top 3 brands by importance
    top_3_brands = brand_features.nlargest(3, 'importance')
    
    # Sum the importance of the remaining 'Brand' features
    other_brands_importance = brand_features[~brand_features['feature'].isin(top_3_brands['feature'])]['importance'].sum()
    
    # Create the 'Brand (summed)' row with only the remaining 'Other' brand importance
    brand_row = pd.DataFrame({'feature': ['Brand (summed)'], 'importance': [other_brands_importance]})
    
    # Create a DataFrame for the top 3 brands and their importance
    top_3_brands = pd.concat([top_3_brands, brand_row])
    
    # Filter for 'Ingredient' columns and sum importance for ingredient features
    ingredient_features = importance_df[importance_df['feature'].str.contains('Ingredient')]
    ingredient_importance_sum = ingredient_features['importance'].sum()
    
    top_3_ingredients = ingredient_features.nlargest(3, 'importance')
    
    # Remove the one-hot encoded 'Brand' and 'Ingredient' features from the DataFrame
    importance_df = importance_df[~importance_df['feature'].str.contains('Brand|Ingredient')]
    
    # Create new DataFrame rows for the summed 'Ingredient' importance
    ingredient_row = pd.DataFrame({'feature': ['Ingredient (summed)'], 'importance': [ingredient_importance_sum]})
    
    # Concatenate the new rows with the existing DataFrame
    importance_df = pd.concat([importance_df, top_3_brands, ingredient_row], ignore_index=True)
    
    # Sort the features by importance
    importance_df = sorted_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Define colors for the top 3 brands
    top_3_colors = ['rgba(255, 0, 0, 0.6)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 255, 0.6)']
    
    # Plotly stacked bar chart
    fig = go.Figure()
    
    # Add bars for the top 3 brands
    for i, (index, row) in enumerate(top_3_brands[top_3_brands['feature'] != 'Brand (summed)'].iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Brand (summed)'],
            orientation='h',
            name=row['feature'],
            marker_color=top_3_colors[i % len(top_3_colors)]  # Use modulo to avoid index out of range
        ))
    
    # Add bar for 'Other' brands
    fig.add_trace(go.Bar(
        x=[other_brands_importance],
        y=['Brand (summed)'],
        orientation='h',
        name='Other',
        marker_color='rgba(0, 0, 0, 0.8)',  # Gray color for 'Other'
        text='Other Brands',
        textposition='inside'
    ))
    
    # Add bars for the top 3 ingredients
    top_3_colors_ingredient = ['rgba(0, 0, 139, 0.6)', 'rgba(255, 140, 0, 0.6)', 'rgba(0, 100, 0, 0.6)']
    for i, (index, row) in enumerate(top_3_ingredients.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Ingredient (summed)'],
            orientation='h',
            name=row['feature'],
            textposition='inside',
            marker_color=top_3_colors_ingredient[i % len(top_3_colors_ingredient)]  # Use modulo to avoid index out of range
        ))
    
    # Add the summed "Ingredient (summed)" bar
    fig.add_trace(go.Bar(
        x=[ingredient_importance_sum],
        y=['Ingredient (summed)'],
        orientation='h',
        name='Ingredient',
        marker_color='rgba(0, 0, 0, 0.8)',  # Primary color for Ingredient summed
        text='Other Ingredients',
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Plot the remaining features
    fig.add_trace(go.Bar(
        x=importance_df[~importance_df['feature'].str.contains('Brand|Ingredient')]['importance'],
        y=importance_df[~importance_df['feature'].str.contains('Brand|Ingredient')]['feature'],
        orientation='h',
        name='Other Features',
        marker_color='rgb(255, 204, 204, 0.1)'  # Gray color for other features
    ))
    
    # Update layout for better display
    fig.update_layout(
        title="Feature Importances in Random Forest (Including Summed Categories)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        barmode='stack',  # Stack bars on top of each other
        yaxis={'categoryorder': 'total ascending'}
    )   

    return fig
