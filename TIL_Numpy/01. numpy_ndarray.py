#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[12]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# ### array
# - 많은 숫자 데이터를 하나의 변수에 넣고 관리 할 때 리스트는 속도가 느리고 메모리를 많이 차지하는 단점이 있다. 
# - 배열(array)을 사용하면 적은 메모리로 많은 데이터를 빠르게 처리할 수 있다. 
# - 배열은 리스트와 비슷하지만 _다음과 같은 점_에서 다르다.
#     - 모든 원소가 같은 자료형이어야 한다.
#     - 원소의 갯수를 바꿀 수 없다.
#     - 넘파이의 array 함수에 리스트를 넣으면 ndarray 클래스 객체 즉, 배열로 변환해 준다. 
#     - np.array([값1, 값2, ...]) -> 값을 리스트 형태로 넘겨줘야 함

# In[3]:


x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(x, type(x))
print(y, type(y))


# In[4]:


plt.plot(x,y)


# ### 다양한 방법으로 ndarray 생성하기

# 1. np.array() 로 생성하기

# In[5]:


x = np.array([1,2,3,4])
print(x)


# In[7]:


y = np.array([[1,2,3,4], [5,6,7,8]])
print(y)


# 2. np.arange() 함수로 생성하기

# In[8]:


np.arange(10)


# In[9]:


np.arange(1,10)


# In[10]:


np.arange(1,10,2)


# In[11]:


np.arange(5,101,5)


# 3. np.ones(), np.zeros() 로 array 생성하기

# In[13]:


np.ones(4) # array의 원소값을 1로 초기화
np.zeros(4) # array의 원소값을 0으로 초기화


# In[15]:


np.ones((3,4))  # 3행 4열의 2차원 array 배열 생성
np.zeros((2,3)) # 2행 3열의 2차원 array 배열 생성


# 4. np.empty([행, 렬], dtype = 배열타입), np.full() 로 생성하기
# 
# - numpy.empty 함수는 (값의 초기화를 수행하지 않고) 주어진 형태와 타입을 갖는 새로운 array를 반환
#     - 초기화되지 않으므로 수행 속도가 빠름 - 값을 반드시 재정의 해 줘야 함(출력되는 값의 의미 없는 값)
#     - 바로 생성하고 다른 의미 있는 값을 넣어줄 때 주로 사용함
# - np.full((행, 렬), 초기화 값)

# In[16]:


a = np.empty([2,2]) # 어떤 메모리를 잡았느냐에 따라 값이 다르게 나타날수도 있음
b = np.empty([2,2], dtype=int)
a
b


# In[17]:


np.full((3,4),7)


# 5. np.eye(n, k=m) 로 생성하기
# 
# - 단위 행렬 생성: 대각선이 1, 나머지는 0으로 채워지는 행렬
# - n: 행, 열 / k = 1|0|-1

# In[18]:


np.eye(3) # k 값을 생략했기 때문에 k = 0으로 자동으로 만들어짐


# 6. np.linspace 로 생성하기
# - numpy.linspace(start, stop, num=50)
# - 지정된 간격 동안 균일한 간격의 숫자를 반환
# 
# 반환 num구간 [위에 계산 균등 샘플 start, stop].

# In[20]:


np.linspace(1,10,3)
np.linspace(1,10,4)
np.linspace(1,10,5)


# #### reshape 함수 활용
#  - ndarray의 형태, 차원을 바꾸기 위해 사용

# In[42]:


x = np.arange(1,16)
x
x.shape
x.reshape(3,5)


# ### random 서브 모듈의 함수를 통해 ndarray 생성하기

# #### random 서브 모듈
# 
# - rand 함수
#     - 0,1 사이의 분포로 랜덤한 ndarray 생성

# In[28]:


np.random.rand(4)
np.random.rand(4,5) # 4*5 행렬
a = np.random.rand(4,5,3) # 5*3 행렬을 4개 반환
a[0] # a array의 첫 번째 원소에 해당하는 행렬 출력
a[1]


# #### randn함수
#  - n: normal distribution(정규분포)
#  - 정규분포로 샘플링된 랜덤 ndarray 생성

# In[31]:


np.random.randn(5)
np.random.randn(3,4)
np.random.randn(2,3,4)


# #### randint()
# - 특정 정수 사이에서 랜덤하게 샘플링

# In[33]:


np.random.randint(1,100,size=(5,))
np.random.randint(1,100,size=(5,3))


# In[ ]:


#### seed() 함수
- 랜덤한 값을 동일하게 다시 생성하고자 할 때 사용


# In[37]:


np.random.seed(23)
np.random.randn(3, 4)


# #### 확률 분포에 따른 ndarray 생성 함수
# - uniform : 이산형 정규 분포
# - normal : 정규분포(일반)
# - 등등...

# In[38]:


np.random.uniform(1.0,3.0, size=(4,5))


# In[39]:


np.random.normal(1.0,3.0, size=(4,5))
np.random.randn(3,4)


# #### np.random.choice(a, size, replace=Truue|False, p)
# - 배열로부터 임의 표본 추출(random sampling)
# - replace = 복원추출(True)|비복원추출(False)
#     - 복원: 한 번 추출한 원소값을 원본에 도로 넣음
# - p : 모집단의 원소에 대해 발생 확률을 미리 알고 있으면 확률값을 넘겨줄 수 있음

# ![](./choice.png)

# In[ ]:


# 정수가 주어진 경우, np.arange(해당숫자)와 같은 결과 반환
np.random.choice(100, size=(3,4))


# In[ ]:


x = np.array([1, 2, 3, 1.5, 2.6, 4.9])


# In[41]:


np.random.choice(x, size=(2,2), replace=True) # 복원추출
np.random.choice(x, size=(2,2), replace=False) # 비복원추출


# In[ ]:


p = [0.1, 0, 0.3, 0.6, 0]
np.random.choice(5, 3, p=p) # 모집단에서 각 원소가 발생할 확률을 이미 알고 있는 경우 or 특정한 확률로 표본을 추출하고 싶은 경우에 p 사용

