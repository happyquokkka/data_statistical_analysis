#!/usr/bin/env python
# coding: utf-8

# ### 상관분석
# 
# - 목적 : 두 연속형 변수간 어떤 선형관계를 가지는지 파악
# 
# 
# - 귀무가설 : 두 변수는 독립적이다(상관성이 없다) <- pvalue가 높으면 귀무가설 채택
# - 대립가설 : 두 변수는 독립적이지 않다(상관성이 있다) 
# 
# 
# - 산점도로 시각화

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import *
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# In[4]:


df = pd.read_excel('../../data/일별_금은달러.xlsx')
df.head()
df.tail()
len(df)


# In[7]:


# 일자별로 정렬
df.sort_values(by = '일자', inplace=True)
# 인덱스는 안 쓸거니까 따로 reset 하지 않아도 됨


# In[8]:


df.head(1)


# In[16]:


# 산점도 시각화

# 그래프 기본 설정
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (10, 6)

plt.scatter(df['일자'], df['금값'])
plt.scatter(df['일자'], df['은값'])
plt.scatter(df['일자'], df['달러 환율'])
plt.xticks(df['일자'].iloc[::8])


# In[14]:


plt.scatter(df['금값'], df['은값'])


# In[15]:


plt.scatter(df['금값'], df['달러 환율'])


# In[19]:


pd.plotting.scatter_matrix(df.drop('일자', axis=1))


# ## 상관분석(피어슨 상관계수, 스피어만 상관계수) - pvalue를 반환

# - 두 변수 모두 연속형 변수 일때 사용하는 상관계수로 x와 y에 대한 상관 계수는 다음과 같이 정의 됨
# 
#     - 상관계수가 1에 가까울 수록 양의 상관관계가 강하다고 함
#     - 상관계수가 -1에 가까울수록 음의 상관관계가 강하다고 함
#     - 상관계수가 0에 가까울수록 상관관계가 약하다고 함
# 
# ![](./상관계수.png)

# In[20]:


pearsonr(df['금값'],df['은값'])
# 상관계수 : 0.97
# pvalue가 0에 수렴 -> 귀무가설 기각


# In[21]:


pearsonr(df['금값'], df['달러 환율']) # 상관계수 -0.67 -> 음의 상관: 금값이 올라가면 달러 환율은 떨어진다


# In[22]:


pearsonr(df['은값'], df['달러 환율'])# 상관계수 -0.67 -> 음의 상관: 은값이 올라가면 달러 환율은 떨어진다


# #### 스피어만 상관 계수
# 
# 
# - 분석하고자 하는 두 연속형 분포가 심각하게 정규분포를 벗어난다거나 순위척도 자료일 때 사용
#     - 연속형 자료일 때는 각 측정값을 순위 척도 자료로 변환시켜 계산
#     
#     
#     
# - 두 변수 순위 사이의 단조 관련성(한 변수가 증가할 때 다른 변수가 증가하는지 감소하는지에 대한 관계)만을 측정하는 상관계수
# - 선형적인 상관 관계를 나타내지 않는다

# ![](./상관분석함수.png)

# ![](./스피어만통계량.png)

# In[23]:


import itertools #스피어만 상관계수
target_columns = ['금값', '은값', '달러 환율']
for col1, col2 in itertools.combinations(target_columns, 2) : # for문을 이용해서 모든 경우의 수를 생성
    result = spearmanr(df[col1], df[col2])
    print("{} ~ {}: coef:{}, p-value: {}".format(col1, col2, result[0], result[1]))


# In[25]:


for col1, col2 in itertools.combinations(target_columns, 2):
    print(col1,col2)


# ### 상관행렬

# In[26]:


df.drop('일자', axis=1).corr(method='pearson')


# In[27]:


df.drop('일자', axis=1).corr(method='spearman')

