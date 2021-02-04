#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.core.pylabtools import figsize


# In[ ]:


#matplotlib inline
figsize(12, 8)
sns.set()


# In[ ]:


countries = pd.read_csv('countries.csv', decimal=',')


# In[ ]:


new_column_names = [
    'Country', 'Region', 'Population', 'Area', 'Pop_density', 'Coastline_ratio',
    'Net_migration', 'Infant_mortality', 'GDP', 'Literacy', 'Phones_per_1000',
    'Arable', 'Crops', 'Other', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture',
    'Industry', 'Service'
]

countries.columns = new_column_names


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[ ]:


countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# In[ ]:


countries.head(5)


# In[ ]:


countries.info()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[ ]:


def q1():
  return sorted(countries['Region'].unique())
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[ ]:


def q2():
  pop_density = countries['Pop_density'].values.reshape(-1, 1)
  est = KBinsDiscretizer(n_bins=10, encode='ordinal')
  est.fit(pop_density)
  return int((pop_density > est.bin_edges_[0][-2]).sum())
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[ ]:


countries['Climate'] = countries['Climate'].fillna(0)


# In[ ]:


def q3():
  return countries['Region'].nunique() + countries['Climate'].nunique()
q3()


# alternativamente, posso usar `OneHotEncoder` de `sklearn.preprocessing`

# In[ ]:


def q3_version2():
  encoder = OneHotEncoder()
  old_columns = countries[['Region', 'Climate']].values
  new_columns = encoder.fit_transform(old_columns)
  feature_names = encoder.get_feature_names(['Region', 'Climate'])
  return feature_names.size
q3_version2()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[ ]:


pipeline = Pipeline([
  ('median_imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
  ('standard_scaler', StandardScaler())
])


# In[ ]:


countries_subset = countries.select_dtypes(include=['int64', 'float64'])
pipeline.fit_transform(countries_subset);


# In[ ]:


sample = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[ ]:


def q4():
  sample_df = pd.DataFrame([sample], columns=new_column_names)
  sample_subset = sample_df.select_dtypes(include=['int64', 'float64'])
  sample_subset = pipeline.transform(sample_subset)
  return round(float(sample_subset[0, 9]), 3)
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[ ]:


def q5():
  q1, q3 = countries['Net_migration'].quantile([.25, .75]).values
  inf_limit = q1 - 1.5 * (q3 - q1)
  sup_limit = q3 + 1.5 * (q3 - q1)
  inf_outliers = np.sum(countries['Net_migration'] < inf_limit)
  sup_outliers = np.sum(countries['Net_migration'] > sup_limit)
  return int(inf_outliers), int(sup_outliers), False
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[ ]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset='train', categories=categories, 
                               shuffle=True, random_state=42)


# In[ ]:


def q6():
  vectorizer  = CountVectorizer()
  X = vectorizer.fit_transform(newsgroup.data)
  keys = vectorizer.get_feature_names()
  phone_index = keys.index('phone')
  phone_count = X[:, phone_index].sum()
  return int(phone_count)
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


def q7():
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(newsgroup.data)
  keys = vectorizer.get_feature_names()
  phone_index = keys.index('phone')
  phone_count = X[:, phone_index].sum()
  return round(float(phone_count), 3)
q7()

