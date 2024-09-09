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
  
  # Aplicar a função à coluna alvo
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

  return best_model, importance_df
