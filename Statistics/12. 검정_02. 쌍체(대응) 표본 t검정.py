#!/usr/bin/env python
# coding: utf-8

# #### 쌍체 표본 t검정
# 
# - x의 변화에 따라 y의 값이 결정
# - x에 변화를 가했을 때 그 x가 변화해서 추출된 값을 검정(효과가 있는지)
#     - 예시 : A반 학생들이 보충수업 후 국어 시험을 봤을 때 보충 수업의 효과가 있는지를 확인하는 것
#     
#     
# - 가설설정 -> 데이터 정규성 검정 -> T-test -> 결론 도출

# #### 선행조건
# 
# - 실험전(x)과 실험후(y)의 데이터는 정규분포를 따르지 않아도 됨
# - 측정값의 차이(x와 y의 차이)는 정규성을 갖고 있어야 함
# 
# ![](./대응t정리.png)

# ![](./대응t량.png)

# In[1]:


import pandas as pd
from scipy.stats import *


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# In[3]:


df = pd.read_csv("../../data/다이어트약_효과검증.csv")


# In[4]:


df.columns


# In[5]:


df.head()
len(df)


# In[6]:


before = df['다이어트전(kg)']
after = df['다이어트후(kg)']


# In[7]:


# 정규성 검정 : 두 값의 차이가 정규분포를 따르는지 확인
kstest(after-before,'norm')

# pvalue가 거의 0에 가까운 수치(지수가 - 면 소수 이하로 내려감)


# In[8]:


# from scipy.stats imort shapiro : 정규성 검정 함수

# 귀무가설 : 데이터는 정규분포이다(정규성을 갖는다)
# 대립가설 : 데이터는 정규분포가 아니다

shapiro(after-before)

# pvalue가 0.05보다 크므로 귀무가설 채택


# #### 대응(쌍체)표본  t 검정 함수
# 
# - scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagate', alternative='two-sided')
#     - nan_policy {‘propagate’:nan을 반환, ‘raise’:오류발생, ‘omit’:무시} : NaN 처리 방법
#     - alternative='two-sided'|'less'|'greater' : 양측검정|왼쪽단측|오른쪽단측

# In[9]:


# 귀무가설 : 대응표본 두 집단의 평균은 같다 -> 다이어트 약의 효과는 없다는 의미
# 대립가설 : 대응표본 두 집단의 평균은 다르다 -> 다이어트 약의 효과는 있다
# pvalue가 0에 가까우므로 귀무가설은 기각
# 통계량은 확실한 양의 효과가 있다 - 다이어트약의 효과가 있다

ttest_rel(before, after)


# ### 개별실습 : 마케팅에 따른 판매액 차이 : htest02d.csv

# In[12]:


df = pd.read_csv("../../data/htest02d.csv")
df.head()
df.tail()


# In[13]:


df.info()


# In[16]:


before = df['before'].values
after = df['after'].values
before # 마케팅 이전
after # 마케팅 이후


# In[17]:


# 정규성 검정 : 두 값의 차이가 정규분포를 따르는지 확인
kstest(after-before,'norm')

# pvalue가 거의 0에 가깝다 -> 두 값의 차이는 정규분포를 따름 


# In[18]:


# 귀무가설 : 대응표본 두 집단의 평균은 같다 -> 마케팅 효과가 없다
# 대립가설 : 대응표본 두 집단의 평균은 다르다 -> 마케팅 효과가 있다

ttest_rel(before, after)

# 통계량을 확인했을 때 판매액은 증가했다는 것을 알 수 있음 
# (통계량 = 음수 -> before의 평균이 after의 평균보다 작음)
# pvalue가 유의수준 0.05보다 작으므로 귀무가설 기각 -> 마케팅 효과가 있다

