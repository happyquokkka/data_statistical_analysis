#!/usr/bin/env python
# coding: utf-8

# # 추측통계의 기본
# 
# - 추측통계 : __일부 데이터로 전체의 통계적 성질을 추측하는 것__
# 
# ![](./추측통계예시.png)

# ## 모집단과 표본
# 
# 
# - 모집단(population) 
#     - 조사 대상이 되는 관측 가능한 개체로 된 집단 전체(관심이 있는 집단)
#     
#     
# - 표본(sample)
#     - 모집단에서 선택된 모집단 구성단위의 일부
# 
# ![](./모집단과표본.jpg)

# ### 표본의 추출 방법
# - 무작위 추출(임의 추출) : 임의로 표본을 추출하는 방법
# - 복원 추출 : 여러 차례 동일한 표본이 선택될 수 있는 방법
# - 비복원 추출 : 동일한 표본은 한 번만 선택되는 방법

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


df = pd.read_csv('../data/ch4_scores400.csv')
scores = np.array(df['score'])
scores[:10]


# In[18]:


# 복원 추출
np.random.choice([1,2,3],3)


# In[21]:


# 비복원 추출
np.random.choice([1,2,3],3, replace=False)


# In[ ]:


np.random.seed(0) # 매번 동일한 샘플 추출하기 위한 기본 값 설정
np.random.choice([1,2,3],3)


# - 모집단(scores)에서 sample 추출

# In[22]:


np.random.seed(0)
sample = np.random.choice(scores, 20)

# 표본 평균
sample.mean()


# In[23]:


# 전체 데이터의 평균
scores.mean()


# In[24]:


for i in range(5):
    sample = np.random.choice(scores, 20)
    print(f'{i+1}번째 무작위 추출로 얻은 표본평균', sample.mean())

# 표본평균은 70점 전후이지만 나름대로 산포도도 나타내고 있음


# ### 확률의 기본
# 
# - 확률(probability)
#     - 어떤 사건(event)이 일어날 가능성
#     - 예. 
#         - 동전 던지기에서 앞면이 나올 가능성
#         - 주사위 던지기에서 특정 눈금이 나올 확률
#         - 주식투자에서 이득을 볼 가능성
#         
#         
# 
# - 의사결정
#     - 확실성(certainty) 보다는 불확실한 상황에서 이루어지는 경우가 많음
#     - 미래에 대한 불확실성의 정도가 더욱 심하다고 할 수 있음 
#     - 불확실성 하에서 의사결정의 오류를 줄이기 위해 확률에 대한 연구 필요

# ### 확률변수 : 취하는 값과 그 값이 나올 확률이 결정되어 있는 것
#     - 일정한 확률로 나타나는(발생하는) 사건에 대해 숫자를 부여한 변수
#     - ex. 주사위를 굴리면 1-6사이의 숫자가 약 1/6 확률로 발생
# 
# - 시행
#     - 확률 변수의 결과를 관측하는 것
#     - 주사위를 굴리면 어떤 숫자가 나올지 모르지만 숫자는 하나씩 확정
#     
#     
# - 실현값
#     - 시행에 의해 관측되는 값
#     - 주사위를 굴려서 나온 값
#     
#     
# - 사건
#     - 시행 결과로 나타날 수 있는 일
#     - 눈은 1 또는 눈은 홀수 등
#     - 주사위의 눈은 확률 변수 X
#     - 눈이 1이 되는 사건의 확률 P(x=1)=1/6
#     - 눈이 홀수인 사건의 확률 P((X=1)U(X=3)U(X=5) = 1/6+1/6+1/6 = 1/2
# 
#     - 근원사건
#         - 세부적으로 더 분해할 수 없는 사건
#         
#         
# - 상호배반 : 동시에 일어날 수 없는 사건
#     - 눈이 1 또는 2 또는 3’이라는 사건과 ’눈이 6’이라는 사건
# 

# ## 확률분포
# 
# - 확률변수가 어떻게 움직이는지를 나타낸 것
# 
# ![](./확률분포.png)
# ![](./확률분포2.png)

# ### 주사위 예제

# - 공정한 주사위
#  - 주사위의 무게 중심이 정확하게 중심에 있는 주사위
# 
# ![](./표4-1.jpg)

# - 불공정한 주사위
#     - 주사위의 무게 중심이 중심에 있지 않아 특정한 눈이 더 잘 나오는 편향된 주사위
#     - 그런 주사위가 있다고 해도 불공정하다는 것은 알지만 각 눈이 어느 정도의 확률로 나오는 건지는 알 수 없음
#         - 주사위를 몇 번 굴려서 그 눈의 비율을 확률분포로 추출하는 것이 일반적
#         - 이 주사위는 아래 표의 확률 분포를 따르고, 특정한 눈이 나오기 쉬운 주사위라고 가정함
# 
# ![](./표4-2.jpg)

# In[8]:


# 불공정한 주사위의 확률 분포를 구하는 실험

# 확률 변수에 필요한 변수가 취할 수 있는 값
dice = [1,2,3,4,5,6]

# 상태공간의 각 값이 나올 수 있는 확률 : prob
prob = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]


# - np.random.choice(a, size=None, replace=True, p=None)
# 
#     - a : 배열이면 배열 자체, 정수이면 arange(a)명령으로 배열 생성
#     - size : 정수, 샘플 숫자
#     - replace : boolean. True이면 복원추출, False면 비복원추출
#     - p : 배열, 각 데이터가 선택될 수 있는 확률

# In[41]:


np.random.choice(dice, p=prob) # 불공정한 주사위


# In[42]:


### 불공정한 주사위로 표본 선택

num_trial = 100
sample = np.random.choice(dice, num_trial, p=prob)
sample


# In[43]:


freq,_ = np.histogram(sample, bins=6, range=(1,7))


# In[44]:


freq


# In[ ]:


# 정해 놓은 확률과 일치하는지 확인
pd.DataFrame({'frequency':freq,
              'relative frequency':freq / num_trial},
              index = pd.Index(np.arange(1, 7), name='dice'))


# In[45]:


# 100번의 시도는 정해져 있는 확률과는 차이가 있다

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
# 실제의 확률분포를 가로선으로 표시
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
# 막대 그래프의 [1.5, 2.5, ..., 6.5]에 눈금을 표시
ax.set_xticks(np.linspace(1.5, 6.5, 6))
# 주사위 눈의 값은 [1, 2, 3, 4, 5, 6]
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('dice')
ax.set_ylabel('relative frequency')
plt.show()


# #### 10000번 시도

# In[46]:


num_trial = 10000
sample = np.random.choice(dice, size=num_trial, p=prob)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
# 실제의 확률분포를 가로선으로 표시
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
# 막대 그래프의 [1.5, 2.5, ..., 6.5]에 눈금을 표시
ax.set_xticks(np.linspace(1.5, 6.5, 6))
# 주사위 눈의 값은 [1, 2, 3, 4, 5, 6]
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('dice')
ax.set_ylabel('relative frequency')
plt.show()


# ## 추측통계와 확률
# 
# ### 전교생의 시험점수 분포를 확인하기 위해 계급폭을 1점으로 하여 히스토그램 작성

# In[51]:


# scores


# In[53]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(scores, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
plt.show()


# In[52]:


# 점수 68점은 4%
count = 0

for i in scores :
    if i == 68 :
        count += 1
        
print(len(scores))

print(count)

count/len(scores) * 100


# In[88]:


np.random.choice(scores)


# In[89]:


# 무작위 추출을 했을 때 68점의 확률을 확인
sample = np.random.choice(scores, 10000)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
plt.show()


# In[91]:


# sample 데이터에서 점수 68점은 
# 4% 로 모집단 전체에서 68점의 비율과 비슷한 수치
count = 0

for i in sample :
    if i == 68 :
        count += 1
        
print(len(sample))

print(sample)

count/len(sample) * 100


# ### 표본평균
# 
# - 표본 하나하나가 확률변수이므로 표본들의 평균으로 계산되는 표본 평균도 확률변수
# 

# In[93]:


# 무작위 추출로 표본 크기가 20인 표본을 추출하여 표본평균을 계산하는 작업을 10000번 수행 후 그 결과를 히스토그램으로 표현

sample_means = [np.random.choice(scores, 20).mean()
                for _ in range(10000)]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample_means, bins=100, range=(0, 100), density=True)
# 모평균을 세로선으로 표시
ax.vlines(np.mean(scores), 0, 1, 'gray')
ax.set_xlim(50, 90)
ax.set_ylim(0, 0.13)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
plt.show()


# - 가운데 세로선이 모평균
# - 표본평균은 모평균을 중심으로 분류되는 것을 확인
#     - 무작위 추출에 의한 표본 평균으로 모평균을 추측할 수 있다는 근거가 됨
