#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[ ]:


#%matplotlib inline

from IPython.core.pylabtools import figsize

figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[ ]:


np.random.seed(42)
    
df_rv = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                   "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[ ]:


# Sua análise da parte 1 começa aqui.


# In[ ]:


df_rv.head()


# In[ ]:


df_rv.describe()


# In[ ]:


ax = df_rv.plot.hist(bins=32, alpha=0.6)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quartis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quartis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[ ]:


def q1():
    df1 = df_rv.quantile([.25, .50, .75])
    return tuple(round(i, 3) for i in df1.normal - df1.binomial)


# In[ ]:


q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[ ]:


def q2():
    q = ECDF(df_rv['normal'])
    mean = df_rv['normal'].mean()
    std = df_rv['normal'].std()
    p = q([mean - std, mean + std])
    return float(round(p[1] - p[0], 3))


# In[ ]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[ ]:


def q3():
    m_binom = df_rv['binomial'].mean()
    v_binom = df_rv['binomial'].std()
    m_norm = df_rv['normal'].mean()
    v_norm = df_rv['normal'].std()
    dff_m = round(m_binom - m_norm, 3)
    dff_v = round(v_binom - v_norm, 3)
    return (dff_m, dff_v)


# In[ ]:


q3()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[ ]:


df_stars = pd.read_csv("pulsar_stars.csv")

df_stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(df_stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

df_stars.loc[:, "target"] = df_stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[ ]:


# Sua análise da parte 2 começa aqui.


# In[ ]:


df_stars.head()


# In[ ]:


df_stars.describe()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


def q4():
    x = df_stars['mean_profile']
    x = x[df_stars['target'] == False]
    x = (x - x.mean()) / x.std()
    false_pulsar_mean_profile_standardized = x
    normal_quantiles = sct.norm.ppf([0.80, 0.90, 0.95])
    false_pulsar_quantiles = ECDF(x)([0.80, 0.90, 0.95])
    return tuple(round(i, 3) for i in false_pulsar_quantiles)


# In[ ]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


def q5():
    x = df_stars['mean_profile']
    x = x[df_stars['target'] == False]
    x = (x - x.mean()) / x.std()
    false_pulsar_mean_profile_standardized = x
    normal_quantiles = sct.norm.ppf([0.25, 0.50, 0.75])
    false_pulsar_quantiles = ECDF(x)([0.25, 0.50, 0.75])
    return tuple(round(i, 3) for i in false_pulsar_quantiles - normal_quantiles)


# In[ ]:


q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
