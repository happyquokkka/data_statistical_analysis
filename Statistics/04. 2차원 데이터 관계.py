#!/usr/bin/env python
# coding: utf-8

# * 교재 3장 정리

# ### 2차원 데이터의 정리
# - 두 데이터 사이의 관계를 나타내는 지표

# In[1]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('precision', '3')
pd.set_option('precision', 3)


# In[4]:


df = pd.read_csv('../data/ch2_scores_em.csv',
                 index_col='student number')


# In[7]:


en_scores = np.array(df['english'])[:10]
ma_scores = np.array(df['mathematics'])[:10]


scores_df = pd.DataFrame({'english':en_scores,
                          'mathematics':ma_scores},
                         index=pd.Index(['A', 'B', 'C', 'D', 'E',
                                         'F', 'G', 'H', 'I', 'J'],
                                        name='student'))
scores_df


# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

plt.scatter(scores_df['english'], scores_df['mathematics'])

plt.xticks([40,45,50,55,60,65,70])
plt.xlabel('english score')
plt.ylabel('mathematics score')
plt.yticks([60,65,70,75,80,85])

for i in range(0, len(scores_df)) :
    plt.text(scores_df['english'][i]+0.5,scores_df['mathematics'][i]+0.5,scores_df.index[i])

# 수직선 : plt.axvline(x, color= ...) : 영어의 평균
plt.axvline(x=scores_df['english'].mean(), color='r', linewidth=1, linestyle=':')
# 수평선 : plt.axhline(y, color = ...)
plt.axhline(y=scores_df['mathematics'].mean(), color='r', linewidth=1, linestyle=':')


# ## 共分散(공분산)
# 
# - 보통 Cov 라고 표현한다. 공분산은 두 개 또는 그 이상의 랜덤 변수에 대한 의존성을 의미
# 
# - 직사각형의 가로길이는 영어 점수의 편차, 세로는 수학 점수의 편차
# - 공분산은 면적, 음의 면적도 가능(음의 상관)
# - 아래 그림에서 H와 E의 면적은 양의 값(양의 상관관계),  C는 음의 값(음의 상관관계)
# 
# ![](./cov.jpg)

# In[14]:


# 각 과목의 편차 및 과목간 공분산

summary_df = scores_df.copy()
summary_df['english_deviation'] =    summary_df['english'] - summary_df['english'].mean()
summary_df['mathematics_deviation'] =    summary_df['mathematics'] - summary_df['mathematics'].mean()
summary_df['product of deviations'] =    summary_df['english_deviation'] * summary_df['mathematics_deviation']
summary_df


# In[15]:


summary_df['product of deviations'].mean()
# 공분산의 평균이 62.8 이므로 영어와 수학은 양의 상관관계에 있다고 볼 수 있음
# (영어 잘하면 수학도 잘하고 영어를 못하면 수학도 못함)


# #### numpy의 공분산 함수 : cov(data1, data2, ddof=)
# - 반환값 : 공분산 행렬
# - 반환 행렬 중 [0,1] 과 [1,0] 의 원소가 공분산 값에 해당됨
# 
# \begin{pmatrix}
#   {영어,영어} & {영어,수학} \\
#   {수학,영어} & {수학,수학}
# \end{pmatrix}
# 

# In[16]:


en_scores
ma_scores


# In[18]:


cov_mat = np.cov(en_scores, ma_scores, ddof=0)
cov_mat


# #### 각 과목의 분산값

# In[19]:


cov_mat[0,0], cov_mat[1,1]


# In[21]:


np.var(en_scores, ddof=0), np.var(ma_scores, ddof=0)


# ### 상관계수
# 
# 
# - 공분산의 단위는 직감적으로 이해하기 어려우므로, 단위에 의존하지 않는 상관을 나타내는 지표
#     - 시험 점수간의 공분산 (점수X점수), 키와 점수간의 공분산 (cm X 점수)
# - 상관계수는 공분산을 각 데이터의 표준편차로 나누어 단위에 의존하지 않음
# 
# ![](./상관계수수식.png)
# 
# 
# - 양의 상관은 1에 가까워지고, 음의 상관은 -1에 가까워짐. 
# - 서로 상관이 없으면 0

# In[23]:


# 수식으로 계산
np.cov(en_scores, ma_scores, ddof=0)[0,1]/ (np.std(en_scores) * np.std(ma_scores))
# 두 변수의 공분산 / 두 변수의 표준편차의 곱


# #### 상관계수 함수(np.corrcoef(x,y) / df.corr())

# In[25]:


np.corrcoef(en_scores, ma_scores)
# 수학점수와 영어점수의 상관계수 : 0.819


# In[26]:


# 데이터프레임에 적용
scores_df.corr()


# - 양의 상관은 1에 가까워지고, 음의 상관은 -1에 가까워지고, 무상관은 0
# - 상관계수가 -1일 때와 1일 때 데이터는 완전히 직선상에 놓임

# <img src='상관계수예시.jpg' width=500 height=500>

# ## 2차원 데이터의 시각화

# ### 산점도

# In[27]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


english_scores = np.array(df['english'])
math_scores = np.array(df['mathematics'])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
# 산점도
ax.scatter(english_scores, math_scores)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')

plt.show()


# ### 회귀 직선
# 
# ![](./회귀직선수식.png)

# In[31]:


# 기울기와 절편
poly_fit = np.polyfit(english_scores, math_scores,1)

poly_1d = np.poly1d(poly_fit)

# 직선을 그리기 위해 x좌표를 생성
xs = np.linspace(english_scores.min(), english_scores.max())

# xs에 대응하는 y좌표를 구함
ys=poly_1d(xs)


# In[32]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.scatter(english_scores, math_scores, label='score')
ax.plot(xs, ys, color='gray',
        label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
# 범례의 표시
ax.legend(loc='upper left')

plt.show()


# ### 히트맵
# 
# - 히스토그램의 2차원 버전으로 색을 이용해 표현하는 그래프
# - 영어 점수 35점부터 80점, 수학 점수 55점부터 95점까지 5점 간격

# In[33]:


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

c = ax.hist2d(english_scores, math_scores,
              bins=[9, 8], range=[(35, 80), (55, 95)])

c[3]
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.set_xticks(c[1])
ax.set_yticks(c[2])

# 컬러 바의 표시
fig.colorbar(c[3],ax=ax)
plt.show()

