import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

def get_deodo_data():
  df = pd.read_csv('deodorants_final.csv')
  
  # Define rating_cat as customer satisfaction (1 for yes, 0 for no)
  def converter_categoria(valor):
      if valor <= 4:
          return '0'
      else:
          return '1'
  
  # Aplicar a função à coluna alvo
  df['rating_cat'] = df['rating'].apply(converter_categoria)
  
  df.drop(columns=['rating_number'], inplace=True)
  
  # Lista das colunas a serem removidas
  cols_to_drop = ['main_category', 'average_rating',
                  'price', 'images_x', 'videos', 'store',
                  'bought_together', 'images_y',
                  'user_id', 'helpful_vote', 'verified_purchase']
  
  # Remover as colunas do DataFrame
  df = df.drop(columns=cols_to_drop)
  
  
  top_50_brands = brand_counts.head(50)
  
  
  # Calcular a contagem de cada marca
  brand_counts = df['Brand'].value_counts()
  
  # Selecionar as 50 marcas mais frequentes
  top_50_brands = brand_counts.head(50)
  
  # Filtrar o DataFrame para manter apenas as linhas com essas marcas
  df_filtered = df[df['Brand'].isin(top_50_brands.index)]
  
  
  
  top_50_scent = scent_counts.head(50)
  
  
  # Calcular a contagem de cada 'Scent' no DataFrame filtrado
  scent_counts = df_filtered['Scent'].value_counts()
  
  # Selecionar os 50 'Scent' mais frequentes
  top_50_scent = scent_counts.head(50)
  
  # Filtrar o DataFrame para manter apenas as linhas com esses 'Scent'
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
  
  # Aplicar a função de padronização ao DataFrame
  df_filtered_scent['Item Form'] = df_filtered_scent['Item Form'].apply(standardize_item_form)
  
  # Contar as ocorrências de cada 'Item Form'
  item_form_counts = df_filtered_scent['Item Form'].value_counts()
  
  # Manter apenas as categorias com pelo menos 20 ocorrências
  top_item_forms = item_form_counts[item_form_counts >= 20].index
  df_filtered_scent = df_filtered_scent[df_filtered_scent['Item Form'].isin(top_item_forms)]
  
  
  
  Item_Form_counts = df_filtered_scent['Item Form'].value_counts()
  Item_Form_counts.head(50)
  
  df_final = df_filtered_scent
  
  df_encoded = df_final.copy()
  
  # Inicializar o LabelEncoder
  encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' remove uma categoria para evitar multicolinearidade
  
  # Aplicar o OneHotEncoder nas colunas categóricas
  encoded_columns = encoder.fit_transform(df[['Item Form', 'Brand', 'Scent']])
  
  # Obter os nomes das novas colunas após a codificação
  encoded_column_names = encoder.get_feature_names_out(['Item Form', 'Brand', 'Scent'])
  
  # Criar um DataFrame com as colunas codificadas
  df_encoded = pd.DataFrame(encoded_columns, columns=encoded_column_names)
  
  # Concatenar com o DataFrame original (removendo as colunas originais categóricas, se necessário)
  df_final = pd.concat([df.drop(columns=['Item Form', 'Brand', 'Scent']), df_encoded], axis=1)
  
  # Exibir o DataFrame codificado
  df_final.columns[:50]
  X = df_final.drop(['title_x','features','description', 'categories', 'details',
       'parent_asin', 'rating', 'title_y', 'text', 'asin','timestamp', 'rating_cat'], axis=1)

  y = df_final['rating_cat']
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  best_params = {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 4}

  
  return df_final, X_train, y_train

def deodo_model(X_train, y_train)
    
  best_params = {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 4}
  clf = DecisionTreeClassifier(**best_params, random_state=42)
  # Treinando o modelo com os melhores parâmetros
  clf.fit(X_train, y_train)
  return clf
