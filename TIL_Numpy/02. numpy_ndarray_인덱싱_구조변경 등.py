#!/usr/bin/env python
# coding: utf-8

# ### 인덱싱과 슬라이싱

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# #### 인덱싱
#  - 파이썬 리스트와 동일한 개념으로 사용
#  - ,를 사용하여 각 차원의 인덱스에 접근 가능
# 

# - 1차원 벡터 인덱싱

# In[3]:


x = np.arange(10)
x


# In[4]:


x[3]=100
x


# - 2차원 행렬 인덱싱

# In[5]:


x = np.arange(10).reshape(2,5)
x


# In[7]:


x[1,1] # 1행의 1열 (인덱스는 zero-based)


# In[8]:


x[0] # 0 행 추출


# #### 슬라이싱
#  - 리스트, 문자열 slicing과 동일한 개념으로 사용
#  - ,를 사용하여 각 차원 별로 슬라이싱 가능
# 

# In[10]:


x=np.arange(10)
x


# In[11]:


x[1:]


# In[15]:


x=np.arange(10).reshape(2,5)
x


# In[17]:


x[0, :2]
x[:1,:2]
x[:1,2]


# ### ndarray shape 변경하기
# #### ravel, np.ravel
#   - 다차원배열을 1차원으로 변경
#   - 'order' 파라미터
#     - 'C' - row 우선 변경
#     - 'F - column 우선 변경
#   - 원본을 참조 함

# In[18]:


x = np.arange(15).reshape(3, 5)
print(x)


# In[19]:


np.ravel(x,order='F') # 열 우선
# array는 인덱스가 달라지기 때문에 배치 순서가 중요함
np.ravel(x,order='c') # 행 우선


# In[20]:


# 원본을 참조함 : 원본인 x도 변경함
# temp 나 x 변수 모두 동일한 주소를 참조하기 때문
temp = x.ravel() # 행 우선 변경이 기본값임
temp


# In[22]:


temp[0] = 100
temp
x


# #### flatten
#  - 다차원 배열을 1차원으로 변경
#  - ravel과의 차이점: copy를 생성하여 변경함(즉 원본 데이터가 아닌 복사본을 반환) -> 아예 메모리를 따로 참조함
#  - 'order' 파라미터
#    - 'C' - row 우선 변경
#    - 'F - column 우선 변경

# In[23]:


y = np.arange(15).reshape(3,5)
print(y)


# In[24]:


t2 = y.flatten(order='F')
t2


# In[25]:


t2[0] = 100
t2
y


# #### reshape 함수
#  - array의 shape을 다른 차원으로 변경
#  - __주의할점은 reshape한 후의 결과의 전체 원소 개수와 이전 개수가 같아야 가능__
#  - 사용 예) 이미지 데이터 벡터화 - 이미지는 기본적으로 2차원 혹은 3차원(RGB)이나 트레이닝을 위해 1차원으로 변경하여 사용 됨
# 

# In[26]:


x = np.arange(36)
print(x)
print(x.shape)
print(x.ndim) # 차원 수 출력


# In[27]:


y = x.reshape(6,6)
y
y.shape
y.ndim


# In[28]:


a = np.arange(1,9)
a
b = a.reshape(2,2,2)
b
b.shape
b.ndim


# In[30]:


k = x.reshape(3,3,-1) #  행을 결정하고 열을 -1로 주면 원소의 개수에 따라 열 개수가 정해짐
k
k.shape
k.ndim


# In[31]:


k = x.reshape(3,-1,3) 
k
k.shape
k.ndim


# In[32]:


x_1 = np.arange(35)
x_1


# In[33]:


x_1.reshape(3,3,4)  # cannot reshape array of size 35 into shape (3,3,4) -> 원소의 개수가 모자라서 생성 불가


# In[34]:


x_1.reshape(3,2,2)  # cannot reshape array of size 35 into shape (3,2,2) -> 원소가 너무 많아서 생성 불가

