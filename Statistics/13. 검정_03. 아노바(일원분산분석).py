#!/usr/bin/env python
# coding: utf-8

# ### 일원분산분석 개요
# 
# - 목적 : 셋 이상의 그룹 간 차이가 존재하는지를 확인하기 위한 가설 검정 방법
# - 영가설(귀무가설) : 세 그룹의 표본평균은 같다
# - 대립가설 : 최소한 한 개 그룹은 차이를 보인다

# ### 선행조건
# - 독립성 : 모든 그룹은 서로 독립적이어야 한다
# 
# 
# - 정규성 : 데이터는 정규분포를 따라야 함
#     - 만약, 정규성을 띄지 않으면 비모수적인 방법인 부호검정 을 진행
#    
#    
# - 등분산성 : 그룹의 데이터에 대한 분산이 같아야 함
#     - Levene의 등분산 검정 : p-value가 0.05 미만이면 분산이 다르다고 판단
#     
#     
# - 분산이 같은지 다른지에 따라 사용하는 통계량이 달라지므로, 비모수적인 방법을 수행해야 함
# 
# 
# ![](./일원통계.png)

# ![](./일원분산절차.png)

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import *

get_ipython().run_line_magic('precision', '3')
np.random.seed(1111)


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# In[7]:


# 데이터 불러오기
df = pd.read_csv("../../data/지점별_일별판매량.csv", engine = "python", encoding='euc-kr')
df.head()


# - 각 지점별 판매량의 평균 간 차이가 있는지 확인
# - 변수(그룹)가 2개를 초과하므로 아노바분석(일원분산분석)시행
# 
# 
# - 귀무가설 : 모든 그룹의 평균이 같다
# - 대립가설 : 최소한 한 개의 그룹이라도 평균이 다르다
# 
# 
# #### 지점별 7월 판매량 간 유의미한 차이가 있는가?

# - 모든 집단에 대해 정규성 검정 후 정규성이 없으면 다른 분석 활용
# - 모든 집단에 대해 등분산 검정 후 분산이 같지 않으면 다른 분석
# 
# - ===>Kruskal-Wallis H Test를 수행해야 함(비모수적인 방법)

# In[8]:


# 데이터 분할 (결측치 제거)
A = df['지점A'].dropna().values
B = df['지점B'].dropna().values
C = df['지점C'].dropna().values


# In[10]:


# 박스 플롯으로 시각화
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (10, 6)

plt.boxplot([A,B,C])

plt.xticks([1,2,3],['지점A','지점B','지점C'])


# In[12]:


# 정규성 검정
# pvalue가 유의수준보다 작으므로 정규분포를 띤다

kstest(A,'norm')
kstest(B,'norm')
kstest(C,'norm')


# In[14]:


# 등분산성 검정
levene(A,B,C)

# pvalue가 0.05보다 크므로 귀무가설 채택(결론: 세 그룹의 분산은 동일하다)


# In[15]:


f_oneway(A,B,C)

# pvalue가 거의 0에 수렴 -> A, B, C 평균에서 최소 어느 한 집단은 유의미한 차이가 존재함


# In[25]:


# 사후 분석

from statsmodels.stats.multicomp import pairwise_tukeyhsd

Data = A.tolist() + B.tolist() + C.tolist() # 배열이 아니라 리스트로 변환해야 함
Group = ['A']*len(A) + ['B']*len(B) + ['C']*len(C)
len(Data), len(Group)

posthoc = pairwise_tukeyhsd(Data,Group)

print(posthoc) # 결과를 보려면 print() 사용해야 함
# reject 결과를 확인 :
# [A,B] | C 로 데이터가 구분됨
# 세 변수에서 C는 다른 집단과 유의미한 차이가 존재함

# 집단 간 평균의 차이 시각화
fig = posthoc.plot_simultaneous()

