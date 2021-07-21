#!/usr/bin/env python
# coding: utf-8

# ### 확률 변수
# - X는 여러 가지 값을 취할 가능성이 있다는 뜻에서는 하나의 변수이지만, 어떤 값을 어느 정도의 가능성으로 취하는가는 거기에 정해진 확률에 의해 나타내게 됨
#     - 이러한 변수를 확률변수라고 함

# ### 확률 분포
# - 확률변수가 어떻게 움직이는지를 나타낸 것

# ### 확률 변수의 구분
# 
# - 이산 확률 변수 : 변수가 취할 수 있는 값의 개수가 유한
#     
# 
# - 연속 확률 변수 : 변수가 취할 수 있는 값의 개수가 무한
# 
# 
# ![](./이산연속확률변수.png)

# ### 확률함수
# - 확률변수 X가 특정 실수 값 x를 취할 확률을 X의 함수(f)로 나타낸 것
#     - 확률질량함수(probability mass function: pmf)
#         - 확률변수가 이산형인 경우
#     - 확률밀도함수(probability density function: pdf)
#         - 확률변수가 연속형인 경우

# ### 확률분포의 평균(mean) - 기댓값
# 
# 
# -  확률 변수의 기대값({E})은 각 사건이 벌어졌을 때의 이득과 그 사건이 벌어질 확률을 곱한 것을 전체 사건에 대해 합한 값이다. 
#     - 어떤 확률적 사건에 대한 평균의 의미로 생각할 수 있다.
# 
#     - E(X) 또는 μX 로 표시
#     - 이산확률분포의 기대값 : 확률을 가중값으로 사용한 가중평균
#     - 연속확률분포의 기대값 : 적분개념의 면적
# 
# ![](./기대값.png)
# 
# 
# - 모평균
#     - 모 평균(population mean) μ는 모 집단의 평균이다. 모두 더한 후 전체 데이터 수 n으로 나눈다.

# ### 이산형 확률 분포
# * 1차원 이산형 확률 분포

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# #### 이산형 확률
# - 확률변수 x가 취할 수 있는 값의 집합 {x_1, x_2, ..., x_k}
# - x가 x_k 라는 값을 취하는 확률
# 
# ![](./이산확률.png)
# 
# - 확률 질량함수(확률함수) = pmf
# 
# 
# ![](./pmf.png)

# ## 불공정한 주사위
# 
# ![](./표4-2.jpg)

# *  위 불공정한 주사위 확률분포의 확률 변수 확인
# 
# ![](./불공정한주사위확률변수.png)

# In[22]:


# 위 식을 함수로 구현

def f(x) :
    if x in x_set :
        return x/21
    else :
        return 0


# In[23]:


# 확률 변수가 취할 수 있는 값의 set
x_set = np.array([1,2,3,4,5,6])


# In[24]:


# 확률변수 x
X = [x_set, f] # 확률분포[x_set,f]에 의해 확률변수 x의 동작이 결정됨


# In[25]:


# 확률 p_k를 구한다
prob = np.array([f(x_k) for x_k in x_set])
x_set
prob

# x_k 와 p_k의 대응을 사전식으로 표시
dict(zip(x_set, prob))


# * dict(zip(x_set, prob)) 은 다음 그림과 같다
# 
# ![](./표4-2.jpg)

# In[9]:


# 이산형 확률분포 그래프
fig = plt.figure(figsize=(10,16))
ax = fig.add_subplot(111)
ax.bar(x_set, prob)
ax.set_xlabel('value')
ax.set_ylabel('probability')

plt.show()


# ### 이산형 확률분포 성질
# 
# - 모든 확률은 0보다 크거나 같아야 하고
# - 확률의 합은 1이다
# 
# ![](./이산확률성질.png)

# In[10]:


prob


# In[11]:


np.sum(prob)
np.all(prob >= 0)


# ### 누적분포함수(분포함수) F(x)
# - x가 x 이하가 될 때의 확률을 반환하는 함수
# - 누적분포함수(cdf)는 주어진 확률변수가 특정 값보다 작거나 같은 확률을 나타내는 함수
# 
# ![](./이산누적.png)

# In[15]:


# 작거나 같은이므로 주어진 x보다 작거나 같은 동안의 확률을 모두 더함 - f(x) (확률함수)를 이용
def F(x) :
    return np.sum([f(x_k) for x_k in x_set if x_k <= x])

np.sum(F(3)) # 주사위 눈이 3 이하가 될 확률


# In[17]:


# 주사위의 눈이 3 이하가 될 확률
F(3) # x_set의 원소값이 3보다 작거나 같은 때까지의 확률의 합계
f(3) # x_set의 원소값이 3일 때 확률


# ### 확률변수의 변환
# 
# - 확률변수에 연산을 적용시켜 변화시킨다고 가정 -> 새로운 데이터 집합 -> 확률변수가 됨
# 
# 
# - 확률변수의 변환 연산 : 2X + 3
# - 위 연산을 적용시켜 변환된 확률변수를 Y라고 한다면

# In[20]:


y_set = np.array([2*x_k + 3 for x_k in x_set])

prob = np.array([f(x_k) for x_k in x_set])

dict(zip(y_set, prob))


# ### 1차원 이산형 확률변수의 지표

# - 확률변수의 평균 : 기댓값
#     - 확률변수를 무제산 시행하여 얻은 실험값의 합산
#     
# ![](./기대값1.png)

# - 수식을 함수로 구현
# 
# ![](./기대값함수구현.png)
# 
# 
# - 인수 g가 확률 변수에 대한 연산을 구현한 함수임
#     - g에 아무것도 지정하지 않으면 확률 변수 x의 기댓값이 구해짐
#     
#     
# - 기댓값: 어떤 사건에 대해 평균적으로 나타날 값
#     - ex) 각 사건이 벌어졌을 때의 이득 * 그 사건이 벌어질 확률 을 전체 사건에 대해 합한 값
#         -  주사위를 한 번 던졌을때, 각 눈의 값이 나올 확률은 1/6이고, 주사위 값은 각 눈금이기 때문에 기댓값은 1*1/6 + 2*1/6 + ~ +6 * 1/6 = 3.5
# 

# #### 불공정 주사위 확률에 대한 기댓값

# In[26]:


# 불공정 주사위에 대한 확률 변수

X


# In[27]:


def f(x):
    if x in x_set:
        return x / 21
    else:
        return 0
    
x_set = np.array([1, 2, 3, 4, 5, 6])


# In[29]:


[x_k * f(x_k) for x_k in x_set]


# In[28]:


np.sum([x_k * f(x_k) for x_k in x_set])


# In[30]:


prob


# In[31]:


# 기댓값 확인
1*0.048 + 2*0.095 + 3*0.143 + 4*0.19 + 5*0.238 + 6*0.286


# In[33]:


sample = np.random.choice(x_set, int(1e6), p=prob)
np.mean(sample)


# In[35]:


# 기댓값 함수 구현
def E(x, g=lambda x:x) :
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


# In[36]:


E(X)


# In[43]:


E(X, g=lambda x: 2*x + 3)


# In[47]:


2*E(X) + 3
# 변환변수를 직접 넣지 않고 기댓값에 같은 연산을 해도 변환변수의 기대값과 동일함


# ### 분산
# 
# - 확률변수의 각 값에서 기대값을 뺀 편차의 제곱을 계산한 후 기대값으로 계산
# 
# 
# ![](./이산분산.png)

# ### 불공정한 주사위의 분산

# In[38]:


### 불공정한 주사위의 기대값 함수 E(X)

mean = E(X)
np.sum([(x_k - mean) **2 * f(x_k) for x_k in x_set])


# In[40]:


# 분산식을 함수로 구현

def V(X, g=lambda x: x) :
    x_set, f = X
    mean = E(X,g) # g는 없어도 되지만 변환변수에 대한 연산을 하기 위해 인수로 생성한 것임
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])


# In[41]:


V(X)


# - 확률변수 X 에 대한 변환변수 2X+3에 대한 분산 계산

# In[42]:


V(X, lambda x: 2*x+3)


# ![](./분산의공식.png)

# In[44]:


2**2 * V(X)

