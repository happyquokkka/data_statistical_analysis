#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import *
import pandas as pd


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# ### 데이터가 얼마나 퍼져 있는지 : 변이통계량
# - 데이터들이 얼마나 흩어져 있는가를 나타내는 것 (산포도)
# - 하나의 수치로 데이터가 흩어진 정도를 계산해서 표현
# - 대표값과 더불어 데이터를 비교하는 경우에 유용하게 사용
#     - ex. 평균이 같은 A집단과 B집단의 성적 : 두 집단이 동일한 집단인가?
#     
# ![](./score.png)
# 
# 
# - 어느 대학에서 같은 과목을 두 교수가 가르친다고 하자.두 교수 모두 평균 C학점을 학생들에게 준다면 그 과목을 배우려는 학생들은 어떤 교수를 선택해도 마찬가지라고 생각할 것이다.
# - 그러나 한 교수는 대부분의 학생들이 평범하다고 생각하여 C만 주고 다른 교수는 학생들이 반은 우수하고 반은 공부를 안 한다고 생각하여 A를 주거나 D-만 준다.
# - 그러므로 이러한 흩어짐의 정보 없이 학생들이 평균 성적 C라는 사실만 가지고 교수를 선택한다면 학점 때문에 어려움에 처할 수도 있게 된다.
# 

# #### 범위(range)
# 
# - 데이터의 최대값과 최소값의 차이
# - 데이터가 퍼져 있는 정도를 나타내는 가장 간단한 방법
# - 범위가 클수록 산포가 크다고 말할 수 있지만
# - 평균과 마찬가지로 극단적인 값에 영향을 받음
# - 데이터 중 2개의 정보(최대값, 최소값)만을 이용하므로 적절한 척도로 사용하기 어려움
# 
# **범위(R)=최댓값 - 최솟값**

# * 중간범위
#     - 최대값과 최소값의 평균

# #### 사분위간 범위 (interquartile range: IQR)
# 
# - 데이터를 크기순서로 나열한 다음, 개수로 4등분할 때 첫 번째 사분위수(Q1:1사분위수, 25%지점)와 세 번째 사분위수(Q3:3사분위수, 75%지점)의 차이
# 
# 
# #### 사분위수 편차(quartile deviation)
# 
# - 범위의 문제점을 보완한 척도
# - 사분위간 범위의 값을 2로 나눈 값으로 사분위 범위의 평균값
# 
# ![](./iqr_variation.png)
# 

# ### 분산(variance)
# 
# - 산포도의 척도로 가장 많이 사용되는 방법
# - 데이터가 퍼져있는 정도의 기준으로 평균을 사용
# 
# 
# - 계산 방법
#     - 각 데이터와 평균의 차이를 제곱하여 합한 값의 평균
# 
# ![](./variance.png)
# 
#     - 데이터가 모집단 전체일 경우에는 데이터의 개수(n)로 나누어 줌
#     - 표본일 경우 (n-1)로 나누어 줌
#     - 표본의 경우 n으로 나누어 주는 것보다 (n-1)로 나누어 주는 것이 더 좋은 척도가 되기 때문인데 표본의 크기가 큰 경우에는 별 차이가 없음

# - 분산 계산 : var(data, ddof=0|1) 함수 사용 ㅡ ddof 생략하면 0

# In[3]:


## ddof인수 : (자유도-모수집단이냐 표본이냐)는 값을 1로 두고 사용한다고 생각하면 편함
## 특별한 경우 제외하고는 모두 sample 데이터이므로 분모를 n-1로 둠
## 즉 , ddof는 1로 둔다

x = [1, 2, 3, 4, 5]

np.var(x, ddof=1) # 분모 n-1
np.var(x) # 분모 n
np.array(x).var()
pd.Series(x).var(ddof=0)

# 값의 스케일에 크게 영향을 받으므로
# 변수를 스케일링(변수를 같은 단위로 맞춤)한 후 분산
# 혹은 표준편차를 활용


# ### 표준편차(standard deviation)
# 
# - 계산된 분산의 제곱근으로 계산
# - __평균을 중심으로 일정한 거리에 포함된 데이터의 비율이 얼마인가를 계산__
# - 모든 데이터를 고려한 척도
# 
# 
# - 특징
#     - 모든 데이터가 동일한 값을 갖는다면 분산과 표준편차는 0으로 계산
#     - 모든 데이터에 동일한 값을 더해 주거나 빼도 변하지 않음
#     - 모든 데이터에 동일한 값(C)을 곱하면 분산은 $분산×C^2$으로 표준편차는 $표준편차×C$ 만큼 커짐
# 
# ![](./standard_ex.png)

# #### 표준편차 계산 : std() 함수 사용
# 
# - 분산에서 제곱의 영향을 없앤 지표
# - 분산과 표준편차가 크면 클수록 산포가 크다

# In[6]:


x = [1, 2, 3, 4, 5]

np.std(x, ddof=1)
np.array(x).std(ddof=0)
pd.Series(x).std(ddof=1)


# #### 변동계수(CV: Coeeficient of Variable) 
# 
# - 표본 표준편차를 표본 평균으로 나눈 값 또는 그 값에 100을 곱한 값
# - 상대 표준편차라고도 함
# - 서로 다른 평균과 표준편차를 갖는 여러 데이터의 흩어진 정도(=산포도)를 비교할 때 주로 사용
# 
# - 변동계수 값이 크다는 것은 데이터가 흩어진 정도가 상대적으로 크다는 의미
# 
# - 표본 변동계수 $ 𝐶𝑉=\frac{S}{\overline{x}}$,  모변동계수  $𝐶𝑉= \frac{𝜎}{𝜇}$
# 
# ![](./변동계수.PNG)

# #### 변동계수 계산 함수: variation(data)
# - 모든 계수가 양수가 아니면 잘 쓰이지 않음
# - 자유도는 0으로 세팅되어 있음

# #### 변동계수가 필요한 이유
# 
# - 표준편차를 구할 때 스케일에 영향 받지 않기 위해 변수를 스케일링 한 후 편차를 구함
#     - 데이터가 모두 양수인 경우: 변동계수 사용

# In[8]:


x1 = np.array([1,2,3,4,5])
x2 = x1 * 10
x2


# In[10]:


np.std(x1, ddof=1)
np.std(x2, ddof=1) # 표준편차는 *10만큼 차이남


# In[11]:


variation(x1)
variation(x2)
# 변동계수는 동일하게 나타남


# In[12]:


# 변동계수 수식

np.std(x1, ddof=1)/np.mean(x1)
np.std(x2, ddof=1)/np.mean(x2)


# #### 스케일링 (표준화)
# 
# - 평균: 0으로 표준편차=1이 됨
# - 각 값들을 상대적인 값으로 변화시키는 기법
# - 평균이 95점(국어), 30점(수학)인 반에서 
#     - 길동이가 받은 취득점수 90점(국어), 80점(수학)이라면 길동이는 어떤 과목을 더 잘한 것인가?
#     
# - 표준화된 데이터는 점수와 같은 단위를 사용하지 않는다

# In[13]:


x1
x2


# - standard scaling : 평균이 0, 표준편차가 1

# In[14]:


z1 = (x1- x1.mean())/x1.std()
z2 = (x2- x2.mean())/x2.std()

z1
z2
# 보통 -3 ~ +3 까지 분포됨


# In[16]:


z1.mean()
z2.mean()
z1.std(ddof=0)


# - MinMax 스케일링

# In[18]:


z1 = (x1 - x1.min()) / (x1.max() - x1.min())
z2 = (x2 - x2.min()) / (x2.max() - x2.min())

x1
x2
print(z1)
print(z2)


# ## 표준화 예제

# In[19]:


df = pd.read_csv('../data/ch2_scores_em.csv',
                 index_col='student number')
# df의 처음 5행을 표시
df.head()


# In[20]:


scores = df.loc[1:10]
scores.index=['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J']
scores.index.name='students'


# In[21]:


scores
scores.mean()
scores.std()


# In[22]:


# np.set_printoptions(precision=20, suppress=True)
# pd.options.display.float_format = '{:.2f}'.format
# import numpy as np
# # numpy float 출력옵션 변경
# np.set_printoptions(precision=3)

import numpy as np
# numpy float 출력옵션 변경
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})


# In[24]:


# 스탠다드 스케일링
em_z = (scores['english']-scores['english'].mean())/scores['english'].std()
mt_z = (scores['mathematics']-scores['mathematics'].mean())/scores['mathematics'].std()
em_z
mt_z


# In[25]:


em_z.mean()
mt_z.mean()
em_z.std()
mt_z.std()


# ### sklearn을 이용한 스케일링 - 머신러닝을 위해 쓰는 경우가 대부분
# - df의 각 열에 대해서 스케일링하는데 적합함

# In[31]:


X = pd.DataFrame({"X1":[1, 2, 3, 4, 5],
    "X2": [10, 20, 30, 40, 50]})

X


# In[32]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # 생성자 함수 이용해 객체 인스턴스 생성
Z = scaler.fit_transform(X) # ndarray로 결과 반환
pd.DataFrame(Z)


# ### 편차값
# 
# - 학력 등의 검사 결과가 집단의 평균값에서 어느 정도 떨어져 있는가를 수치로 나타낸 것. 편차를 표준 편차로 나눠 10배를 한 뒤 50을 더한 것임.
#     - 평균을 50, 표준편차가 10이 되도록 정규화한 값 
# 

# In[33]:


scores


# In[36]:


z = 50 + 10*(scores-np.mean(scores))/np.std(scores)
z
z.mean()
z.std()


# In[38]:


result=pd.concat([scores,z], axis=1)
result.columns=['영어','수학','영어편찻값','수학편찻값']
result[['영어','영어편찻값','수학','수학편찻값']]

