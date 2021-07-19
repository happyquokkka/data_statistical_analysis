#!/usr/bin/env python
# coding: utf-8

# ## 연산의 브로드캐스팅

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# #### 브로드캐스팅
#   - Shape이 같은 두 ndarray에 대한 연산은 각 원소별로 진행
#   - 연산되는 두 ndarray가 다른 Shape을 갖는 경우 브로드 캐스팅(Shape을 맞춤) 후 진행

# #### 브로드캐스팅 Rule
#  - [공식문서](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules)
#  - 뒷 차원에서 부터 비교하여 Shape이 같거나, 차원 중 값이 1인 것이 존재하면 가능
# 

# ![](./브로드캐스팅.jpg)
#     - 출처: https://www.tutorialspoint.com/numpy/images/array.jpg

# In[4]:


x = np.arange(15).reshape(3, 5)
y = np.random.rand(15).reshape(3, 5)
print(x)
print(y)


# * shape이 같은 경우의 연산

# In[5]:


x * y


# - Scalar(상수)와의 연산

# In[6]:


x % 2 == 0


# - shape이 다른 경우 연산
# 

# In[9]:


a = np.arange(12).reshape(4, 3)
b = np.arange(100, 103)
c = np.arange(1000, 1004)
d = b.reshape(1, 3)

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
a
b
c
d


# In[10]:


a + b
# b array의 브로드캐스팅 발생


# In[11]:


# a + c
# operands could not be broadcast together with shapes (4,3) (4,) 
# 한 행에 해당하는 원소의 개수가 같아야만 연산 진행 가능


# In[12]:


a + d


# In[13]:


b_1 = np.arange(100,104)
d = b_1.reshape(4,1)
a
d


# In[14]:


a+d


# ### boolean indexing의 이해

# #### Boolean indexing
#   - ndarry 인덱싱 시, bool 리스트를 전달하여 True인 경우만 필터링

# In[15]:


x = np.random.randint(1, 100, size=10)
print(x)


# In[18]:


x[x % 2 == 0]


# In[19]:


x[x>30]


# ####  다중조건 사용하기
#  - 파이썬 논리 연산자인 and, or, not 키워드 사용 불가
#  - & - AND 
#  - | - OR

# In[20]:


# 짝수이면서 30보다 작은 원소
x[(x % 2 == 0) & (x<30)]


# In[21]:


# 30보다 작거나 50보다 큰 원소
x[(x<30) | (x>50)]


# #### 예제) 2020년 7월 서울 평균기온 데이터
#  - 평균기온이 25도를 넘는 날수는?
#  - 평균기온이 25를 넘는 날의 평균 기온은?

# In[22]:


temp = np.array(
        [23.9, 24.4, 24.1, 25.4, 27.6, 29.7,
         26.7, 25.1, 25.0, 22.7, 21.9, 23.6, 
         24.9, 25.9, 23.8, 24.7, 25.6, 26.9, 
         28.6, 28.0, 25.1, 26.7, 28.1, 26.5, 
         26.3, 25.9, 28.4, 26.1, 27.5, 28.1, 25.8])


# In[23]:


# 전체 수집 일수
len(temp)


# In[ ]:


temp > 25.0


# In[25]:


(temp[temp>25.0])


# In[26]:


len(temp[temp>25.0])


# In[27]:


np.sum(temp>25.0) # True : 1, False : 0


# In[30]:


# 평균기온이 25를 넘는 날의 평균 기온은?
np.mean(temp[temp>25.0])

