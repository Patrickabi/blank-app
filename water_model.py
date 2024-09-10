

def plot_water_feature(df)

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
  
  # Fazer previsões
  y_pred = model.predict(X_test)
  
  # Relatório de classificação
  print(classification_report(y_test, y_pred))
  
  # Importância das características
  feature_importances = model.feature_importances_
  features = X.columns
  
  # Função para obter as 3 maiores importâncias de cada categoria
  def get_top_3_features(category, feature_importances, features):
      category_features = [i for i, f in enumerate(features) if category in f]
      sorted_features = sorted(category_features, key=lambda i: feature_importances[i], reverse=True)[:3]
      return {features[i]: feature_importances[i] for i in sorted_features}
  
  # Obter as 3 maiores características de cada categoria
  top_3_brand = get_top_3_features('Brand', feature_importances, features)
  top_3_material = get_top_3_features('Material', feature_importances, features)
  top_3_color = get_top_3_features('Color', feature_importances, features)
  
  # Somar as importâncias para 'Brand', 'Material', 'Color'
  sum_brand = sum(top_3_brand.values())
  sum_material = sum(top_3_material.values())
  sum_color = sum(top_3_color.values())
  
  # Criar um DataFrame para plotagem, ordenado por importância decrescente
  aggregated_importances = pd.DataFrame({
      'Category': ['Brand', 'Material', 'Color', 'Volume'],
      'Importance': [sum_brand, sum_material, sum_color, feature_importances[features.get_loc('Volume')]]
  }).sort_values(by='Importance', ascending=False)
  
  # Listas de valores para plotagem
  categories = aggregated_importances['Category'].tolist()
  importance_sums = aggregated_importances['Importance'].tolist()
  
  # Listas com as contribuições dos top 3 de cada grupo
  brand_values = list(top_3_brand.values())
  material_values = list(top_3_material.values())
  color_values = list(top_3_color.values())
  
  # Cores para cada subcategoria
  colors_brand = ['#1f77b4', '#aec7e8', '#c6dbef']
  colors_material = ['#ff7f0e', '#ffbb78', '#fdd0a2']
  colors_color = ['#2ca02c', '#98df8a', '#c7e9c0']
  
  # Criar o gráfico
  fig = go.Figure()
  
  # Adicionar barras para 'Brand'
  for i, (feature, value) in enumerate(top_3_brand.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Brand'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_brand[i]),
          showlegend=True
      ))
  
  # Adicionar barras para 'Material'
  for i, (feature, value) in enumerate(top_3_material.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Material'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_material[i]),
          showlegend=True
      ))
  
  # Adicionar barras para 'Color'
  for i, (feature, value) in enumerate(top_3_color.items()):
      fig.add_trace(go.Bar(
          name=feature,
          y=['Color'],
          x=[value],
          orientation='h',
          marker=dict(color=colors_color[i]),
          showlegend=True
      ))
  
  # Adicionar a barra de 'Volume'
  fig.add_trace(go.Bar(
      name='Volume',
      y=['Volume'],
      x=[feature_importances[features.get_loc('Volume')]],
      orientation='h',
      marker=dict(color='lightgrey'),
      showlegend=True
  ))
  
  # Configurar o layout
  fig.update_layout(
      title='Importância das Características Agrupadas e Divididas pelos Top 3',
      xaxis_title='Importância',
      yaxis_title='Categoria',
      barmode='stack',
      template='plotly_white',
      legend_title_text='Top 3 Features'
  )
  
  return fig
