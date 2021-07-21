#!/usr/bin/env python
# coding: utf-8

# ## 대표적인 이산형 확률 분포
# 
# - 베르누이 분포 → 이항분포(binomial distribution)로 확장
# 

# ## 이항분포
# 
# - 이항 분포는 연속된 n번의 독립적 시행에서 각 시행이 확률 p를 가질 때의 이산 확률 분포
# 
# 
# 
# 
# ### 이산 확률 분포
# 
# - 이산 확률 분포(discrete probability distribution)는 이산 확률 변수가 가지는 확률 분포를 의미한다. 여기에서 확률변수가 이산 확률변수라는 말은 확률 변수가 가질 수 있는 값의 개수가 가산 개 있다는 의미

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# In[3]:


# 필요 함수 정의

# 그래프 선의 종류
linestyles = ['-', '--', ':']

# 기대값 구하는 함수
def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])

# 분산 구하는 함수
def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])

# X가 이산형 확률변수인지 확인하는 함수
def check_prob(X):
    x_set, f = X  # 상태공간 = x_set, 확률 만드는 f
    prob = np.array([f(x_k) for x_k in x_set])
    
    # assert문은 if문과 비슷함
    assert np.all(prob >= 0), 'minus probability'
    prob_sum = np.round(np.sum(prob), 6)
    assert prob_sum == 1, f'sum of probability{prob_sum}'
    print(f'expected value {E(X):.4}')
    print(f'variance {(V(X)):.4}')
    
    
    
# 그래프 작성 함수

def plot_prob(X):
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(x_set, prob, label='prob')
    ax.vlines(E(X), 0, 1, label='mean', color='black')
    ax.set_xticks(np.append(x_set, E(X)))
    ax.set_ylim(0, prob.max()*1.2)
    ax.legend()
    
    plt.show()


# ### 베르누이 분포
# 
# - 확률변수가 취할 수 있는 값이 0과 1밖에 없는 분포
#     - 베르누이 분포를 따르는 확률변수의 시행이 베르누이 시행
#     - 1은 성공, 0은 실패
#     - 1이 나오는 확률 p, 0이 나오는 확률 1-p
#     - 파라미터p인 베르누이 분포는 Bern(p)
# ![](./베르누이함수.png)

# #### 동전을 던져서 앞면이 나올 확률
# 
# ![](./동전확률.png)

# #### 주사위를 던져 6이 나오지 않을 확률
# ![](./주사위확률.png)

# #### 베르누이 분포를 함수로 구성

# In[5]:


def Bern(p) :
    x_set = np.array([0,1])
    def f(x) :
        if x in x_set :
            return p ** x * (1-p) ** (1-x)
        else :
            return 0
    return x_set, f


# In[7]:


p=0.3 # 1이 나올 확률이 30%
X = Bern(p)
print(X)


# ![](./베르누이기대값분산.png)

# In[8]:


# 기대값과 분산
check_prob(X)
# 분산은 성공할 확률 * 실패할 확률


# In[9]:


# 기대값과 확률의 관계
plot_prob(X)


# ## scipy.stats 모듈
# 
# - SciPy는 각종 수치 해석 기능을 제공하는 파이썬 패키지
# 
#     - SciPy는 여러개의 서브 패키지로 구성되어 있는데 그 중 scipy.stats 서브패키지는 여러가지 확률 분포 분석을 위한 기능을 제공
#     
# 
# ![](./stats.png)
# 

# In[10]:


p=0.3
# 인수로 확률 파라미터를 취하고 베르누이 분포를 따르는 object를 반환함
# 반환 object가 저장된 변수 rv는 확률변수에 해당함
rv = stats.bernoulli(p)


# In[12]:


# pmf 메서드 : 확률 질량 함수
rv.pmf(0) # 0이 나올 확률 질량 함수
rv.pmf(1)

# 인수를 리스트로 넘길 수 있음
rv.pmf([0,1])


# In[13]:


# cdf 누적 분포 함수
rv.cdf([0,1])


# In[14]:


# 기대값과 분산 계산
rv.mean() # 1일 확률, 즉, 성공할 확률
rv.var()


# ![](./베르누이정리.png)

# ### 이항분포
# - 성공 확률이 μ인 베르누이 시행을 N 번 반복하는 경우
#     - 가장 운이 좋을 때는 N 번 모두 성공
#     - 운이 나쁘면 N 번 모두 실패 (= 한 번도 성공하지 못함)
#     
#     
# - N 번 시행 중 성공한 횟수를 확률 변수 X 라고 한다면 X 의 값은 0 부터 N 까지의 정수 중 하나가 됨
#     - 이런 확률변수를 **이항분포(binomial distribution)**를 따르는 확률변수라고 하며 다음과 같이 표시한다.
# 
# ![](./이항분포.png)
# 
# 
# - X = 성공의 확률이 p 인 베르누이 시행을 N 번 반복할 때 성공의 수가 따르는 분포

# - X = 성공의 확률이 p 인 베르누이 시행을 N 번 반복할 때 성공의 수가 따르는 분포
# - 𝑋가 가질 수 있는 값: 0, 1, 2, …, 𝑛
#     - n번 던져서 x번 성공할 확률
# 
# ![](./이항연산.png)
# 

# ![](./comb.png)
# 
# 
# ##### 위 연산을 실행하는 함수 : comb

# In[17]:


from scipy.special import comb
com = comb(5,2) # n 이 5, k 가 2

(5*4*3*2*1)/(2*1*(3*2*1))


# ![](./이항분포.png)

# In[30]:


## 이항 분포를 따르는 확률변수를 반환하는 함수
from scipy.special import comb

def Bin(n,p) :
    x_set = np.arange(n+1)
    def f(x) :
        if x in x_set :
            return comb(n,x) * p**x * (1-p)**(n-x)
        else :
            return 0
    return x_set, f  # 만들어진 함수 f를 리턴


# In[31]:


n = 10
p = 0.3
X = Bin(n,p)


# In[32]:


X


# In[33]:


check_prob(X)
# 기대값 = 3.0, 분산 = 2.1


# In[34]:


x_set, f = X
x_set
f


# In[35]:


np.array([f(x_k) for x_k in x_set])


# In[38]:


def plot_prob(X) :
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(x_set, prob, label='prob')
    ax.vlines(E(X), 0, 1, label='mean', color='black')
    ax.set_xticks(np.append(x_set, E(X)))
    ax.set_ylim(0, prob.max()*1.2)
    ax.legend()

    plt.show()


# In[39]:


plot_prob(X)


# ### 성공할 확률(p)을 변경해 그래프를 그린 후 비교

# In[44]:


n


# In[48]:


n = 10
linestyles = ['-', '--', ':']
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(n+1)

# 성공확률 0.3, 0.5, 0.7일 때 그래프 비교
for p, ls in zip([0.3, 0.5, 0.7], linestyles) :
    rv = stats.binom(n, p)
    ax.plot(x_set, rv.pmf(x_set), label=f'p:{p}', ls = ls, color='grey')
    

ax.legend()
plt.show()

# 그래프의 모양은 종모양


# In[41]:


list(zip([0.3, 0.5, 0.7], linestyles))


# ### 파이썬 scipy 모듈의 stats.binom 함수 사용하여 이항확률변수 생성
# 
# - stats.binom(시행횟수, 성공확률)

# In[49]:


# 성공확률이 0.6인 베르누이 시행을 10번 반복했을 때의 확률변수 rv를 생성하시오
N = 10
mu = 0.6
rv = stats.binom(N,mu)


# In[51]:


xx = np.arange(N+1) # arange는 stop-1이기 때문에 +1을 해주는 것임
print(xx)
rv.pmf(xx)


# In[54]:


import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':  # 맥OS 
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # 윈도우
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system...  sorry~~~')


# In[55]:


xx = np.arange(N+1)
plt.bar(xx, rv.pmf(xx), align='center')

plt.xlabel("표본값")
plt.ylabel("$P(x)$")
plt.title("이항분포의 확률질량함수")

plt.show()


# ### rvs함수
# 
# - binom.rvs 함수는 이항분포를 따르는 난수를 생성시킴
# 
# - 인자: n, p, size, random_state
#     - random_state 인수는 seed 값 주는 것

# In[56]:


from scipy.stats import binom

binom.rvs(n=50, p=0.5, size=3)


# In[58]:


# rvs 메서드로 무작위 표본을 뽑아내는 시뮬레이션을 한 결과 생성되는 난수는 이항 분포를 따름
np.random.seed(0)
x = rv.rvs(100) # rv(확률변수) 안에 이미 n 과 p 가 설정되어 있으니 size만 넘겨줌
x


# In[59]:


import seaborn as sns
sns.countplot(x)
plt.title("이항분포의 시뮬레이션 결과")
plt.xlabel("표본값")
plt.show()


# #### 포아송 분포
# 
# - 단위 시간 안에 어떤 사건이 몇 번 발생할 것인지를 표현하는 이산 확률 분포
# 
# 
# - 예시
#     - 119 구조대에 걸려오는 시간 당 전화 횟수
#     - 1년 동안 발생하는 진도 4 이상의 지진 횟수
#     - 프러시아 기병 중에서 매년 말에 차여 사망하는 병사의 수
#     - 한 야구경기에서 실책의 수
#     - 서울지역의 5월 중 부도를 낸 중소기업 수
# 
# ![](./포아송.png)
# 

# ### 포아송 분포의 확률함수
# 
# ![](./포아송2.png)

# - 하루에 평균 2건의 교통사고가 발생하는(Poi(2)) 지역에서 하루에 사고가 한 건도 일어나지 않을 확률
# 
# ![](./교통사고.png)

# In[61]:


# 편의상 x_set을 0~19 이하인 정수로 가정
from scipy.special import factorial

def Poi(lam) :
    x_set = np.arange(20)
    def f(x) :
        if x in x_set :
            return np.power(lam,x)/factorial(x) * np.exp(-lam)
        else :
            return 0
    return x_set, f


# In[64]:


X = Poi(10) # 평균 발생 횟수


# In[65]:


x_set, f = X
prob = np.array([f(x_k) for x_k in x_set])


# In[66]:


x_set
prob

