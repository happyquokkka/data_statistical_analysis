#!/usr/bin/env python
# coding: utf-8

# ## 카이제곱 검정 개요
# 
# - 목적 : 두 범주형 변수가 서로 독립적인지 검정
# 
#     
# - 귀무가설 : 두 변수가 서로 독립이다
# - 대립가설 : 두 변수가 서로 종속된다
# 
# - 교차테이블(분할표)로 시각화

# ### 교차 테이블(분할표)
# 
# - 두 변수가 취할 수 있는 값의 조합의 출현 빈도를 나타냄
# 
# ![](./분할표.png)

# ![](./관측값.png)

# ![](./n11기대값.png)

# ![](./카이기대값1.png)

# ![](./카이제곱통계량계산.png)

# ![](./카이수식.png)

# ![](./카이제곱함수.png)

# - 기댓값과 실제값의 차이가 클수록 통계량이 커지게 됨
#     - 통계량이 커질수록 p-value가 감소함
#         - 귀무가설이 기각될 가능성이 높아짐

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import *


# In[5]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# In[6]:


df = pd.read_csv("../../data/htest05.csv", engine = "python") #폐암과 흡연의 연관성 분석
df.head()
df.tail()


# ### 분할표 작성
# - pd.Crosstab()
# - 범주형 변수로 되어있는 요인(factors)별로 교차분석(cross tabulations) 해서, 행, 열 요인 기준 별로 빈도를 세어서 도수분포표(frequency table), 교차표(contingency table) 를 만들어줌
# 
# ![](./데이터재구조화.png)
# 

# In[8]:


# 교차 테이블 생성(분할표)
cross_t = pd.crosstab(df['smoke'], df['disease'])
cross_t
type(cross_t)


# In[9]:


obs = cross_t.values
obs
# 2차원 array


# In[11]:


statistics, pvalue, dof, expected = chi2_contingency(obs, correction=False)
# statistics, pvalue, dof, expected = chi2_contingency(obs)

# 통계량, pvalue, 자유도, 기대값
# 기대도수가 5를 초과하면 false 

pvalue
# 귀무가설 기각 -> 흡연 유무와 폐암 유무는 연관성이 있다


# In[12]:


expected


# In[13]:


# 기대값
pd.DataFrame(expected, columns=cross_t.columns, index=cross_t.index)


# In[14]:


# 카이제곱 통계량
statistics


# ### 성별에 따라 만족도가 달라지는지 - 성별과 만족도는 독립적인지 검정

# In[16]:


# 개별실습

df = pd.read_csv("../../data/성별에따른만족도.csv", engine = "python", encoding='euc-kr')
df.head()


# In[18]:


cross_t = pd.crosstab(df['만족도'], df['성별'])
cross_t


# In[19]:


obs = cross_t.values
obs


# In[25]:


chi2_contingency(obs, correction=False)
statistics, pvalue, dof, expected = chi2_contingency(obs, correction=False)
# statistics, pvalue, dof, expected = chi2_contingency(obs)

# 통계량, pvalue, 자유도, 기대값
# 기대도수가 5를 초과하면 false 

pvalue
# pvalue가 0.018이므로 귀무가설 기각 -> 만족도와 성별은 서로 독립적이지 않다


# In[21]:


expected


# In[22]:


# 기대값
pd.DataFrame(expected, columns=cross_t.columns, index=cross_t.index)


# In[23]:


# 카이제곱 통계량
statistics

