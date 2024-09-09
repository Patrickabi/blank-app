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

def create_df():
    df = pd.read_csv('Base_Baby_18kv3.csv')
    
    #---------------------------------------------------
    # Convert the 'volume' column to numeric, coercing errors to NaN
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Filter rows where 'volume' is between 50 and 2000
    df = df[(df['volume'] >= 50) & (df['volume'] <= 2000)]
    
    # Convert the 'Item_Weight_n' column to numeric, coercing errors to NaN
    df['Item_Weight_n'] = pd.to_numeric(df['Item_Weight_n'], errors='coerce')
    
    # Filter rows where 'Item_Weight_n' is between 1 and 40
    df = df[(df['Item_Weight_n'] >= 1) & (df['Item_Weight_n'] <= 40)]
    # -------------------------------------------------
    
    
    
    # Define rating_cat as customer satisfaction (1 for yes, 0 for no)
    def converter_categoria(valor):
        if valor < 3:
            return '0'
        else:
            return '1'
    
    # Aplicar a função à coluna alvo
    df['rating_cat'] = df['rating'].apply(converter_categoria)
    
    
    # Convert the 'volume' column to numeric, coercing errors to NaN
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Filter rows where 'volume' is between 50 and 2000
    df = df[(df['volume'] >= 50) & (df['volume'] <= 2000)]
    
    # Convert the 'Item_Weight_n' column to numeric, coercing errors to NaN
    df['Item_Weight_n'] = pd.to_numeric(df['Item_Weight_n'], errors='coerce')
    
    # Filter rows where 'Item_Weight_n' is between 1 and 40
    df = df[(df['Item_Weight_n'] >= 1) & (df['Item_Weight_n'] <= 40)]
    
    
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    df_encoded = encoder.fit_transform(df[['Brand', 'Material Type']])
    encoded_columns = encoder.get_feature_names_out(['Brand', 'Material Type'])
    df_encoded = pd.DataFrame(df_encoded, columns=encoded_columns)
    df = pd.concat([df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
    
    
    X = df.drop(columns=['Brand', 'Material Type', 'Size', 'height', 'width', 'thickness', 'Unnamed: 0', 'rating', 'title_review', 'text', 'parent_asin',
    'features', 'description', 'title_product', 'details', 'rating_cat'])
    
    y = df['rating_cat']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return df

df = create_df()

X = df.drop(columns=['Brand', 'Material Type', 'Size', 'height', 'width', 'thickness', 'Unnamed: 0', 'rating', 'title_review', 'text', 'parent_asin',
'features', 'description', 'title_product', 'details', 'rating_cat'])

y = df['rating_cat']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

y_train = y_train.astype(int)
y_test = y_test.astype(int)


def train_rf_model():
    

    rf_model = RandomForestClassifier(
    n_estimators=468,            
    max_depth=3,                 
    min_samples_split=13,         
    max_features='log2',  
    bootstrap=False,              
    min_samples_leaf=5,         
    class_weight=None,           
    random_state=42           
    )


    rf_model.fit(X_train, y_train)

    return rf_model, X_train

def plot_feature_importance(rf_model, X_train):
    
    

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
    
    # Sum the importance of all brand features
    brand_importance_sum = brand_features['importance'].sum()
    
    # Filter for 'Material' columns and sum importance for material features
    material_features = importance_df[importance_df['feature'].str.contains('Material')]
    material_importance_sum = material_features['importance'].sum()
    
    top_3_material = material_features.nlargest(3, 'importance')
    
    # Remove the one-hot encoded 'Brand' and 'Material' features from the DataFrame, but keep Volume and Item_Weight_n
    importance_df = importance_df[~importance_df['feature'].str.contains('Brand|Material') | 
                                  importance_df['feature'].isin(['Volume', 'Item_Weight_n'])]
    
    # Create new DataFrame rows for the summed 'Brand' and 'Material' importances
    brand_row = pd.DataFrame({'feature': ['Brand (summed)'], 'importance': [brand_importance_sum]})
    material_row = pd.DataFrame({'feature': ['Material (summed)'], 'importance': [material_importance_sum]})
    
    # Concatenate the new rows with the existing DataFrame
    importance_df = pd.concat([importance_df, brand_row, material_row], ignore_index=True)
    
    # Sort the features by importance
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Define colors for the top 3 brands
    top_3_colors = ['rgba(255, 0, 0, 0.6)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 255, 0.6)']
    
    # Plotly stacked bar chart
    fig = go.Figure()
    
    # Add bars for the top 3 brands
    for i, (index, row) in enumerate(top_3_brands.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Brand (summed)'],
            orientation='h',
            name=row['feature'],
            marker_color=top_3_colors[i % len(top_3_colors)]  # Use modulo to avoid index out of range
        ))
    
    # Define colors for the top 3 materials
    top_3_colors_material = ['rgba(0, 0, 139, 0.6)', 'rgba(255, 140, 0, 0.6)', 'rgba(0, 100, 0, 0.6)']
    
    # Add bars for the top 3 materials
    for i, (index, row) in enumerate(top_3_material.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Material (summed)'],
            orientation='h',
            name=row['feature'],
            marker_color=top_3_colors_material[i % len(top_3_colors_material)]
        ))
    
    # Add the summed "Brand (summed)" bar
    fig.add_trace(go.Bar(
        x=[0.0],
        y=['Brand (summed)'],
        orientation='h',
        name='Other',
        marker_color='rgba(0, 0, 0, 0.6)',  # Primary color
        text='Other Brand',
        textposition='inside'
    ))
    
    # Add the summed "Material (summed)" bar
    fig.add_trace(go.Bar(
        x=[0.1389],
        y=['Material (summed)'],
        orientation='h',
        name='Other',
        marker_color='rgba(0, 0, 0, 0.6)',  # Primary color
        text='Other Material',
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Keep bars for "Volume" and "Item_Weight_n"
    fig.add_trace(go.Bar(
        x=importance_df[importance_df['feature'].isin(['Volume', 'Item_Weight_n'])]['importance'],
        y=importance_df[importance_df['feature'].isin(['Volume', 'Item_Weight_n'])]['feature'],
        orientation='h',
        name='Volume and Item_Weight_n',
        marker_color='rgba(100, 100, 255, 0.6)'  # Choose a color for Volume and Item_Weight_n
    ))
    
    # Update layout for better display
    fig.update_layout(
        title="Feature Importances in Random Forest (Including Summed Categories)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        barmode='stack',  # Stack bars on top of each other
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Show the figure
    return fig






