#!/usr/bin/env python
# coding: utf-8

# ### numpy에서 자주 사용되는 함수들 정리

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# #### numpy documentation
#  - [numpy 공식 문서 링크](https://www.numpy.org/devdocs/reference/)
#  - numpy에서 제공되는 함수등에 대한 문서
# 

# In[3]:


x = np.arange(15).reshape(3, 5)
y = np.random.rand(15).reshape(3, 5) # 0에서 1 사이의 숫자가 3행 5열로 생성
print(x)
print(y)


# #### 연산 함수
#  - add, subtract, multiply, divide

# In[6]:


np.add(x,y)
np.divide(x,y)
np.multiply(x,y)
np.subtract(x,y)


# In[7]:


x+y


# #### 통계 함수
# - 평균, 분산, 중앙값, 최대/최소값 등등 통계 관련된 함수가 내장

# In[8]:


y


# In[9]:


y.mean() # 평균
y.max()  # 최댓값
y.min()  # 최소값


# In[ ]:


np.var(y) # 분산
np.median(y) # 중간값
np.std(y) # 표준편차


# #### 집계함수
# - sum() : 합계, cumsum() : 누적합계

# In[10]:


x


# In[11]:


np.sum(x, axis=None) # 전체 원소의 합계 
np.sum(x, axis=0)  # 각 열의 합계
np.sum(x, axis=1)  # 각 행의 합계
np.sum(x)


# In[12]:


np.cumsum(x) # 각 원소의 누적 합계


# #### any, all 함수
#  - any: 특정 조건을 만족하는 것이 하나라도 있으면 True, 아니면 False
#  - all: 모든 원소가 특정 조건을 만족한다면 True, 아니면 False
# 

# In[13]:


z = np.random.randn(10)
z


# In[14]:


z>0 


# In[15]:


np.any(z>0) # z 의 원소 중 0을 초과하는 값이 1개라도 존재하면 True 반환


# In[16]:


np.all(z>0) # z 의 원소 모두가 0을 초과하는 값이면 True


# In[17]:


np.all(z != 0)


# #### where(조건, 조건이 참인 경우, 조건이 거짓인 경우) 함수
#  - 조건에 따라 선별적으로 값을 선택 가능
#  - 사용 예) 음수인경우는 0, 나머지는 그대로 값을 쓰는 경우

# In[18]:


z = np.random.randn(10)
z


# In[19]:


np.where(z>0, z, 0)


# In[20]:


np.where(z>0, 10, 0)


# ## axis 파라미터 이해

# #### axis 이해하기
#  - 몇몇 함수에는 axis keyword 파라미터가 존재
#  - axis값이 없는 경우에는 전체 데이터에 대해 적용
#  - axis값이 있는 경우에는, 해당 axis를 **따라서** 연산 적용

# * axis를 파라미터로 갖는 함수를 이용하기
#  - 거의 대부분의 연산 함수들이 axis 파라미터를 사용
#  - 이 경우, 해당 값이 주어졌을 때, 해당 axis를 **따라서** 연산이 적용
#    - 따라서 결과는 해당 axis가 제외된 나머지 차원의 데이터만 남게 됨
#  - 예) np.sum, np.mean, np.any 등등

# In[21]:


x = np.arange(15)
print(x)


# - 1차원 데이터(= 벡터)에 적용

# In[23]:


np.sum(x, axis=0)


# In[25]:


# np.sum(x,axis=1) # axis 1 is out of bounds for array of dimension 1


# - 행렬에 적용(2차원 데이터)

# In[26]:


y = x.reshape(3,5)
y


# In[28]:


np.sum(y, axis=0)
np.sum(y, axis=1)
# np.sum(y, axis=2) # axis 2 is out of bounds for array of dimension 2


# - 3차원 텐서에 적용하기

# In[29]:


nums = np.array(
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
)
nums
nums.ndim


# In[30]:


np.sum(nums, axis=0) # 0행과 0행을 더한 결과 | 1행과 1행을 더한 결과를 새로운 행렬로 출력


# In[31]:


np.sum(nums, axis=1)


# In[32]:


np.sum(nums, axis=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




