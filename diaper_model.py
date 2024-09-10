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
    
    
    X = df.drop(columns=['Brand', 'Material Type', 'Size', 'height', 'width', 'thickness', 'Unnamed: 0', 'Unnamed: 0.1', 'rating', 'title_review', 'text', 'parent_asin',
    'features', 'description', 'title_product', 'details', 'rating_cat'])
    
    y = df['rating_cat']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    
    return df

df = create_df()

X = df.drop(columns=['Brand', 'Material Type', 'Size', 'height', 'width', 'thickness', 'Unnamed: 0','Unnamed: 0.1', 'rating', 'title_review', 'text', 'parent_asin',
'features', 'description', 'title_product', 'details', 'rating_cat'])

y = df['rating_cat']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


y_train = y_train.astype(int)
y_test = y_test.astype(int)
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


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
    
    # Sum the importance of the remaining 'Brand' features
    other_brands_importance = brand_features[~brand_features['feature'].isin(top_3_brands['feature'])]['importance'].sum()
    
    # Create the 'Brand (summed)' row with only the remaining 'Other' brand importance
    brand_row = pd.DataFrame({'feature': ['Brand (summed)'], 'importance': [other_brands_importance]})
    
    # Create a DataFrame for the top 3 brands and their importance
    top_3_brands = pd.concat([top_3_brands, brand_row])
    
    # Filter for 'Material' columns and sum importance for material features
    material_features = importance_df[importance_df['feature'].str.contains('Material')]
    material_importance_sum = material_features['importance'].sum()
    
    top_3_material = material_features.nlargest(3, 'importance')
    
    # Remove the one-hot encoded 'Brand' and 'Material' features from the DataFrame
    importance_df = importance_df[~importance_df['feature'].str.contains('Brand|Material')]
    
    # Create new DataFrame rows for the summed 'Material' importance
    material_row = pd.DataFrame({'feature': ['Material (summed)'], 'importance': [material_importance_sum]})
    
    # Concatenate the new rows with the existing DataFrame
    importance_df = pd.concat([importance_df, top_3_brands, material_row], ignore_index=True)
    
    # Sort the features by importance
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
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
    
    # Add bars for the top 3 materials
    top_3_colors_material = ['rgba(0, 0, 139, 0.6)', 'rgba(255, 140, 0, 0.6)', 'rgba(0, 100, 0, 0.6)']
    for i, (index, row) in enumerate(top_3_material.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Material (summed)'],
            orientation='h',
            name=row['feature'],
            textposition='inside',
            marker_color=top_3_colors_material[i % len(top_3_colors_material)]  # Use modulo to avoid index out of range
        ))
    
    # Add the summed "Material (summed)" bar
    fig.add_trace(go.Bar(
        x=[0.0296551],
        y=['Material (summed)'],
        orientation='h',
        name='Material',
        marker_color='rgba(0, 0, 0, 0.8)',  # Primary color for Material summed
        text='Other Materials',
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Plot the remaining features
    fig.add_trace(go.Bar(
        x=importance_df[~importance_df['feature'].str.contains('Brand|Material')]['importance'],
        y=importance_df[~importance_df['feature'].str.contains('Brand|Material')]['feature'],
        orientation='h',
        name='Other Features',
        marker_color='rgb(255, 204, 204, 0.1)'  # Gray color for other features
    ))
    
    # Update layout for better display
    fig.update_layout(
        title="Feature Importances in Diaper analysis",
        xaxis_title="Importance",
        yaxis_title="Feature",
        barmode='stack',  # Stack bars on top of each other
        yaxis={'categoryorder': 'total ascending'}
    )   

    return fig

def plot_feature_importance1(rf_model, X_train):
    
    importances = rf_model.feature_importances_
    
    # Create a DataFrame for the feature importances and their corresponding feature names
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    })
    
    # Helper function to handle top features and summed categories
    def get_top_features(df, category, n=3):
        category_features = df[df['feature'].str.contains(category)]
        top_features = category_features.nlargest(n, 'importance')
        summed_importance = category_features[~category_features['feature'].isin(top_features['feature'])]['importance'].sum()
        return top_features, summed_importance

    # Get top features and summed importance for 'Brand'
    top_3_brands, other_brands_importance = get_top_features(importance_df, 'Brand')
    # Create the 'Brand (summed)' row with remaining importance
    brand_row = pd.DataFrame({'feature': ['Brand (summed)'], 'importance': [other_brands_importance]})
    
    # Get top features and summed importance for 'Material'
    top_3_material, material_importance_sum = get_top_features(importance_df, 'Material')
    # Create the 'Material (summed)' row with the summed importance
    material_row = pd.DataFrame({'feature': ['Material (summed)'], 'importance': [material_importance_sum]})
    
    # Combine top features and summed rows
    combined_df = pd.concat([
        top_3_brands, brand_row, 
        top_3_material, material_row,
        importance_df[~importance_df['feature'].str.contains('Brand|Material')]
    ], ignore_index=True)
    
    # Sort the features by importance
    combined_df = combined_df.sort_values(by='importance', ascending=False)
    
    # Define colors for the top 3 features
    top_3_colors_brand = ['rgba(255, 0, 0, 0.6)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 255, 0.6)']
    top_3_colors_material = ['rgba(0, 0, 139, 0.6)', 'rgba(255, 140, 0, 0.6)', 'rgba(0, 100, 0, 0.6)']
    
    # Initialize the plotly figure
    fig = go.Figure()
    
    # Add bars for the top 3 brands
    for i, (index, row) in enumerate(top_3_brands[top_3_brands['feature'] != 'Brand (summed)'].iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Brand (summed)'],
            orientation='h',
            name=row['feature'],
            marker_color=top_3_colors_brand[i % len(top_3_colors_brand)]  # Use modulo to avoid index out of range
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
    
    # Add bars for the top 3 materials
    for i, (index, row) in enumerate(top_3_material.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['importance']],
            y=['Material (summed)'],
            orientation='h',
            name=row['feature'],
            textposition='inside',
            marker_color=top_3_colors_material[i % len(top_3_colors_material)]  # Use modulo to avoid index out of range
        ))
    
    # Add the summed "Material (summed)" bar
    fig.add_trace(go.Bar(
        x=[material_importance_sum],
        y=['Material (summed)'],
        orientation='h',
        name='Material',
        marker_color='rgba(0, 0, 0, 0.8)',  # Primary color for Material summed
        text='Other Materials',
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Plot the remaining features
    fig.add_trace(go.Bar(
        x=combined_df[~combined_df['feature'].str.contains('Brand|Material')]['importance'],
        y=combined_df[~combined_df['feature'].str.contains('Brand|Material')]['feature'],
        orientation='h',
        name='Other Features',
        marker_color='rgb(255, 204, 204, 0.1)'  # Gray color for other features
    ))
    
    # Update layout for better display
    fig.update_layout(
        title="Feature Importances in Diaper Analysis",
        xaxis_title="Importance",
        yaxis_title="Feature",
        barmode='stack',  # Stack bars on top of each other
        yaxis={'categoryorder': 'total ascending'}
    )   

    return fig




