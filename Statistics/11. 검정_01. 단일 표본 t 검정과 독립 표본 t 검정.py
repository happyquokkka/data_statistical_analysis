#!/usr/bin/env python
# coding: utf-8

# ### 가설검정단계
# 
# - 가설을 세운다 -> 두 집단의 평균은 같다(귀무가설) vs 두 집단의 평균은 다르다(대립가설)
# - 기준을 세운다 -> 검정통게량을 구한다(유의수준 5%)
# - 결론을 내린다 -> p_value 참고
# 
#     - 기본 양측검정의 결과를 반한하므로 단측 검정으로 해석하려면 p-value/2를 해서 해석
#     - 통계량이 양수인지 음수인지에 따라 해석이 달라진다

# ![](./검정선택.png)

# ### p-value란?
# 
# - 귀무 가설이 참이라고 했을 때 표본 데이터가 수집될 확률
# - P-value가 낮을 수록 대립가설 채택
# - 통상적으로 p-value < 0.05 면 대립가설 채택
# - 이때 0.05를 유의 수준이라고 하며 대게 0.05 또는 0.01 중 선택
# 

# ## T 검정

# ![](./검정종류.png)

# #### 단일표본 t 검정
# 
# - 목적 : 표본그룹의 평균이 기준값과 차이가 있는지를 확인
# - 귀무가설 : 표본평균은 모집단의 평균과 같다
# - 대립가설 : 표본평균은 모집단의 평균과 다르다
# 
#     - 예시: 편의점에서 판매하는 감자튀김의 무게가 130g인지 아닌지를 판단

# #### 선행조건
# 
# - 해당 변수(sample)는 정규분포를 따라야 함 : 정규성 검정이 선행돼야 함
#     - 단, 샘플 수가 많을수록 정규성을 띨 가능성이 높아지므로 샘플 수가 부족한 경우에만 정규성 검정을 수행함
#     - 만약 정규성을 띠지 않으면 비모수적인 방법인 부호검정을 진행

# #### t 통계량
# 
# ![](./t통계량.png)

# #### 정규성 검정 방법 : Kolmogorov-Smornov 검정
# 
# - KS test라 함 
# - 관측한 샘플들이 특정 분포를 따르는지 확인 하기 위한 검정 방법
# - KS test는 특정 분포를 따른다면 나올 것이라 예상되는 값과 실제 값의 차이가 유의한지를 확인하는 방법
#     - 관측한 샘플들이 특정 분포를 따르는지 확인하기 위한 검정방법임
#         - 예상되는 값과 실제값의 차이가 클수록 분포를 따르지 않는다고 보며, 
#         - 차이(pvalue)가 작을 수록 분포를 따른다고 봄
#     - 해당분포를 정규분포로 설정하여 정규성 검정에도 사용

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import *

get_ipython().run_line_magic('precision', '3')
np.random.seed(1111)


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# - map은 리스트의 요소를 지정된 함수로 처리해주는 함수
#     - map은 원본 리스트를 변경하지 않고 새 리스트를 생성
# 

# In[3]:


a = [1.2, 2.5, 3.7, 4.6]
a = list(map(int, a))
a


# In[8]:


# data 불러오기
import os

with open('../../data/성인여성_키_데이터.txt','r') as f :
    data = f.read().split('\n')
    print(type(data[0]))
    data = list(map(float,data)) # float(실수)로 변환
    print(type(data[0]))


# In[9]:


data[:3]
len(data)

# 데이터 개수가 30개 미만이므로 대응표본검정 중 t검정 사용


# In[11]:


# data


# - 수집한 데이터는 경기 지역 성인 여성 25명의 키 데이터다. 경기지역 성인 여성 키의 평균은 163cm이다 라는
#      - 자료에 근거해서 수집한 sample 로 경기 지역의 성인 여성 키의 평균이 163cm 인지 검정을 통해 확인

# #### 데이터가 25개이고 표본이 하나이므로 단일표본 t검정을 수행
# - 정규성을 띠는지 확인
# - 표본이 한 개이므로 분산과는 상관이 없음
#     - ttest_1samp(집단, popmean=기준값) 함수 사용
# - 귀무가설 : 집단의 평균은 모집단의 평균(기준값 163)과 같다

# In[12]:


ttest_1samp(data,163)

# 통계량 : statistic=-2.979804412662668
# p_value=0.006510445335847954 -> 유의수준 0.05보다 작은 값이므로 귀무가설 기각! 
# 귀무가설 : '집단의 평균은 모집단의 평균(기준값)과 같다'
# 통계량이 음수이므로 경기 지역 여성의 평균 키가 163cm 미만으로 추정할 수 있다


# In[13]:


# 정규성 검정
# kstest() : 모든 분포에 대하여 검정할 수 있고, 정규분포는 'norm' 인수로 검정 
# 통계량과 p-value를 반환

kstest(data, 'norm') # p-value가 0.0이고, 0.05보다 작으므로 정규성을 띤다


# - alternative = 'two-sided'|'less'|'greater'
#     - 기본값 : 'two-sided' (양측검정)

# In[14]:


ttest_1samp(data, 163) # 기본 양측 검정을 진행
ttest_1samp(data,163, alternative='two-sided')
ttest_1samp(data,163, alternative='less') # 단측검정 (더 작다)
ttest_1samp(data,163, alternative='greater') # 단측검정 (더 크다)


# #### 독립 표본 t 검정
# - 목적 : 서로 다른 두 집단의 평균 비교
# 
# - 귀무가설 : 두 집단의 평균은 같다
# - 대립가설 : 두 집단의 평균은 차이가 있다
# 
#     - 예시 : 중간고사의 국어 점수 A반, B반의 평균을 비교했을 때 A반의 평균이 3점 높았다
#     - 이 두 반은 국어 점수의 차이가 있는지 확인 해 보자 

# #### 선행조건
# 
# - 독립성 : 두 그룹은 독립적이어야 한다
# 
# 
# - 정규성 : 데이터는 정규분포를 따라야 한다
#     - 만약, 정규성을 띠지 않으면 비모수적인 방법인 부호검정을 진행
# 
# 
# - 등분산성 : 두 그룹의 데이터에 대한 분산이 같아야 함
#     - Levene의 등분산 검정 : p-value가 0.05 미만이면 분산이 다르다고 판단
#     
#     
# - 분산이 같은지 다른지에 따라 사용하는 통계량 계산식이 달라지므로 함수내에서 설정을 달리 해야 함
# 
# ![](./독립t등분산.png)
# ![](./독립표본t이분산.png)

# - 두 반의 국어 공통 평가 점수가 있을 때 두 반의 평균이 유의미한 차이가 있는지 확인(절대적인 차이는 X)

# In[26]:


# df1 = pd.read_csv('../../data/반별_점수_type1.csv', encoding='utf-8')
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb9 in position 0: invalid start byte
df1 = pd.read_csv('../../data/반별_점수_type1.csv', encoding='euc-kr')
df1.head()
df1.tail()

df1.info()


# In[21]:


# df1을 A반과 B반으로 분리
g_A = df1['점수'].loc[df1['반']=='A'].values # 계산하는 데에는 인덱스는 필요 없으므로 값만 array로 반환 
g_B = df1['점수'].loc[df1['반']=='B'].values
g_A, g_A.mean()
g_B, g_B.mean()

# A반과 B반의 인원수는 20명, 10명으로 동일하지 않음


# In[22]:


# 정규성 검정
kstest(g_A, 'norm')
kstest(g_B, 'norm') # pvalue가 0 이므로 정규성을 띤다(= 정규분포를 따른다)


# In[23]:


# 등분산성 검정
levene(g_A, g_B)

# 귀무가설 : 두 집단은 등분산성을 띤다
# pvalue=0.164964086222101 로 유의수준 0.05보다 크므로 귀무가설을 채택


# - 두 집단의 분산 확인 -> 분산 값이 다름 but, 
#     - pvalue와 유의수준을 비교했을 때 등분산성을 띤다고 볼 수 있음

# In[24]:


np.var(g_A, ddof=1), np.var(g_B, ddof=1)


# In[30]:


# 두 집단이 등분산성을 갖고 있으므로 equal_var = True로 설정
# 두 집단이 등분산성을 갖고 있지 않으면 equal_var = False로 설정
# pvalue가 유의수준 0.05보다 작으므로 두 집단의 평균은 차이가 있다
# 통계량이 양수이므로 g_A의 평균이 더 높다

ttest_ind(g_A, g_B, equal_var=True)


# #### scipy.stats.mannwhitneyu(a,b): 정규성검정이 만족하지 않으면 수행

# #### Tip. 다른 데이터 포맷인 경우 확인해야 할 사항

# In[27]:


df2 = pd.read_csv('../../data/반별_점수_type2.csv', engine='python', encoding='euc-kr')
df2.head()
# B반의 데이터는 float이므로 NaN 값이 들어있을 가능성이 있음


# In[28]:


## 수집 샘플의 길이가 달라서 NaN이 포함된 상태로 df가 만들어졌을 수 있으므로 NaN 처리를 해야함
g_A = df2['A반'].dropna().values
g_B = df2['B반'].dropna().values


# - 검정을 위한 데이터에 의도하지 않은 NaN은 제거해야 함 -> 0으로 처리하면 평균이나 분산 등의 기준값이 달라지게 됨
