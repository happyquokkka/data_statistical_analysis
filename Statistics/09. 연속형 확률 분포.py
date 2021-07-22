#!/usr/bin/env python
# coding: utf-8

# # 대표적인 연속형 확률 분포
# 
# - 연속형 확률 분포
#     - 정규분포
#     - 카이제곱분포
#     - t-분포
#     - F-분포
# 
# 
# - 정규분포(normal distribution)
#         - 대표본 모집단의 합, __평균 추론__ 시 활용
#     - t-분포
#         - __소표본 모집단 평균 추론 시__ 활용 (보통 30개)
#         - 선형모형 회귀계수 추론(종속변수 정규분포 가정)시 활용
#     - 카이제곱(χ2)분포
#         - 모집단 분산 추론 시(데이터 정규분포 가정) 활용
#         - 카이제곱 검정 시 활용
#     - F-분포
#         - 두 모집단 분산 차이 비교 시 활용(데이터 정규분포 가정)
#         - 분산분석 시 활용
#         - (설명하는 변동/설명하지 못하는 변동)이 F분포를 따름
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate
from scipy.optimize import minimize_scalar

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


# 한글 문제
# matplotlit의 기본 폰트에서 한글 지원되지 않기 때문에
# matplotlib의 폰트 변경 필요
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


# In[5]:


linestyles = ['-', '--', ':']

def E(X, g=lambda x: x):
    x_range, f = X
    def integrand(x):
        return g(x) * f(x)
    return integrate.quad(integrand, -np.inf, np.inf)[0]

def V(X, g=lambda x: x):
    x_range, f = X
    mean = E(X, g)
    def integrand(x):
        return (g(x) - mean) ** 2 * f(x)
    return integrate.quad(integrand, -np.inf, np.inf)[0]

def check_prob(X):
    x_range, f = X
    f_min = minimize_scalar(f).fun
    assert f_min >= 0, 'density function is minus value'
    prob_sum = np.round(integrate.quad(f, -np.inf, np.inf)[0], 6)
    assert prob_sum == 1, f'sum of probability is {prob_sum}'
    print(f'expected vaue {E(X):.3f}')
    print(f'variance {V(X):.3f}')
    
def plot_prob(X, x_min, x_max):
    x_range, f = X
    def F(x):
        return integrate.quad(f, -np.inf, x)[0]

    xs = np.linspace(x_min, x_max, 100)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(xs, [f(x) for x in xs],
            label='f(x)', color='gray')
    ax.plot(xs, [f(x) for x in xs],
            label='f(x)', color='gray')
    ax.plot(xs, [F(x) for x in xs],
            label='F(x)', ls='--', color='gray')

    ax.legend()
    plt.show()


# In[6]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# ## 정규분포
# 
# - 가장 대표적인 분포
# - 가우시안(Gaussian)분포라고도 불림
# - 연속형 확률변수에 대한 분포
# - 평균(μ)과 표준편차(σ)로 모양이 결정
# - 좌우 대칭, 종모양
#     - 중앙 한 점이 뾰족함
# - 평균 = 중앙값 = 최빈값
# - 자연계, 사회현상에 많이 나타남
# 
# ![](./정규종.png)

# ![](./정규비교.png)

# ### 정규분포의 확률밀도 함수 f(x)
# ![](./정규수식.png)

# * 확률밀도함수
# 
# ![](./정규밀도.png)

# In[8]:


def N(mu, sigma) : # mu: 평균, sigma: 표준편차
    x_range = [-np.inf, np.inf]
    
    def f(x) :
        return 1/np.sqrt(2*np.pi * sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
    return x_range, f


# In[9]:


# 평균이 2이고 표준편차가 0.5인 정규분포를 따르는 확률변수 X 정의
mu = 2
sigma = 0.5
X = N(mu, sigma)


# In[10]:


# 기대값과 분산을 출력해 줌
check_prob(X) 


# In[13]:


# 정규분포를 따르는 확률변수 X 의 그래프
plot_prob(X, 0, 4)

# 실선은 정규분포 그래프, 파선은 누적분포 그래프 - x값이 커질 때 y값이 작아질 수 없음


# - 남자 고등학생 키의 평균이 170cm 이고 표준편차가 5cm 라면 우연히 만난 남자 고등학생의 키는 N(170, 5^2)을 따른다고 할 수 있다
# 
# - 확률변수 = 남자 고등학생의 키 ~ N(170,5^2)
#     - 남자 고등학생의 키는 평균이 170cm, 표준편차가 5cm인 정규분포 N을 따른다
# 
# 
# - 이 사실을 바탕으로 우연히 만난 남자 고등학생의 키가 165cm 이상, 175cm 이하일 확률은?
# 
# 
# ![](./정규고등학생.png)

# In[15]:


mu, sigma = 170, 5
X = N(mu, sigma)

x_range, f = X

np.round(integrate.quad(f, 165, 175)[0],6) # round 함수는 소수점 이하 수 제거해줌


# - 모의고사 평균 점수가 70점이고 표준편차가 8점이라면, 우연히 만난 학생의 점수는 N(70,8^2)를 따른다고 함
# - 모의고사 점수 ~ N(70,8^2)
# 
# 
# - 위 사실을 바탕으로 우연히 만난 학생의 점수가 54점 이상, 86점 이하일 확률은?
# ![](./정규모의고사.png)

# In[16]:


def N(mu, sigma) : # mu: 평균, sigma: 표준편차
    x_range = [-np.inf, np.inf]
    
    def f(x) :
        return 1/np.sqrt(2*np.pi * sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
    return x_range, f


# In[18]:


mu, sigma = 70, 8
X = N(mu, sigma)

x_range, f = X
np.round(integrate.quad(f, 54, 86)[0],6)


# ### scipy.stats 사용해서 확인
# 
# - stats.norm(평균, 표준편차) : 정규분포를 따르는 확률변수를 객체 인스턴스로 반환

# In[19]:


# 기대값이 2이고 표준편차가 0.5인 정규분포를 따르는 확률변수 rv를 생성
rv = stats.norm(2, 0.5)


# In[20]:


# rv 의 확률변수의 평균과 분산을 계산
# mean(), var()
rv.mean()
rv.var()


# In[21]:


# 확률밀도함수 : pdf()
rv.pdf(2)
# x가 2일 때의 y값 (정규분포 그래프의 한 점을 구한 것뿐임)


# In[27]:


## rv 확률변수의 분포를 그래프로 확인
xx = np.linspace(-2, 6, 100)
pdf = rv.pdf(xx)
# xx
# pdf
plt.plot(xx,pdf)

plt.title("확률밀도함수 ")
plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.show()


# - 확률변수의 누적 확률 분포 값 : cdf 메서드로 계산
# - 누적분포함수 : P(X <= x) 를 계산하는 함수
#     - norm.cdf(x)
# 
# ![](./누적정규.png)

# In[28]:


# 확률변수 rv가 1.7보다 작은 값이 되는 확률
rv.cdf(1.7)


# - norm.isf(위치값) : 상위 위치값 % 지점을 반환
# 
# 
# ![](./백프로수식.png)
# ![](./백프로.png)

# In[30]:


xx = np.linspace(-2, 6, 1000)
pdf = rv.pdf(xx)
plt.plot(xx, pdf)


# 위 그래프에서 상위 30% 지점값 반환
rv.isf(0.3)
rv.isf(0.95) # 상위 95% 지점값 


# - norm.interval(구간비율)
#     - norm.interval(0.9) : 90% 구간 지점을 반환 - 왼쪽과 오른족 끝에 5%씩 남김
# 
# ![](./인터발.png)

# In[32]:


xx = np.linspace(-2, 6, 1000)
pdf = rv.pdf(xx)
plt.plot(xx, pdf)
# 90% 구간
rv.interval(0.9)


# In[33]:


# 왼쪽과 오른쪽 끝에서 5%씩 남김
rv.isf(0.95), rv.isf(0.05)


# In[34]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

xs = np.linspace(-5, 5, 100)
params = [(0, 1), (0, 2), (1, 1)]
for param, ls in zip(params, linestyles):
    mu, sigma = param
    rv = stats.norm(mu, sigma)
    ax.plot(xs, rv.pdf(xs),
            label=f'N({mu}, {sigma**2})', ls=ls, color='gray')
ax.legend()

plt.show()


# ### 남자 고등학생의 키 ~ N(170, 5^2) 의 분포를 표현하는 그래프

# In[36]:


rv = stats.norm(170,5)

xx = np.linspace(150,190,100)
pdf = rv.pdf(xx)

plt.plot(xx, pdf)

plt.title("남자고등학생의 키")
plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.show()


# ![](./jeong.png)

# ### 카이제곱분포
# 
# ![](./kai.png)
# 
# 
# 
# - z 는 정규확률변수(평균이 0, 표준편차가 1인 정규분포를 따르는 확률변수), 자유도는 정규분포의 갯수(n)
# 
# 
# ![](./kai2.png)

# In[43]:


n = 10
rv = stats.norm(0,1)

sample_size = int(1e6) # 1000000 개

# 표준 정규분포로부터 10 * 100만 사이즈로 무작위 데이터(난수)를 추출
Za_sample = rv.rvs((n,sample_size))

chi_sample = np.sum(Za_sample**2, axis=0)
chi_sample


# ### 카이제곱 분포를 따르는 확률변수 생성 : stats.chi2(n)

# In[51]:


n = 10
rv_true = stats.chi2(n) # 카이제곱 분포를 따르는 확률변수로 rv_true 지정

xs = np.linspace(0,30,100)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

# sample 데이터 카이제곱 분포 계산 데이터를 이용한 히스토그램
ax.hist(chi_sample, bins=100, density=True, alpha=0.5, label='chi_sample')
ax.plot(xs, rv_true.pdf(xs), label=f'chif{n})',color='grey')
plt.show()


# In[52]:


# 자유도값을 변경해가면서 그래프 모양 확인 ㅡ z는 컴퓨터 내부에서 생성

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

xs = np.linspace(0,20,500)

for n, ls in zip([3, 5, 10], linestyles) :
    rv = stats.chi2(n)
    ax.plot(xs, rv.pdf(xs), label=f'chi({n})', ls=ls, color='grey')    


# In[ ]:


rv = stats.chi2(5) # 자유도가 5일 때
rv.isf(0.05)
rv.isf(0.95)


# ![](./카이정리.png)

# ## t분포 : 정규분포에서 모평균의 구간추정 등에 사용되는 확률분포
# 
# - stats.t
# 
# ![](./t분포.png)

# - 좌우 대칭인 분포
# - 표준 정규분포보다 양쪽 끝이 더 두껍다
# - 자유도가 커지면 표준 정규분포에 가까워진다

# In[54]:


n = 10
rv1 = stats.norm()
rv2 = stats.chi2(n)

sample_size = int(1e6) # 100만개

Z_sample = rv1.rvs(sample_size)
Y_sample = rv2.rvs(sample_size)

t_sample = Z_sample / np.sqrt(Y_sample/n)

len(t_sample)


# In[59]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# t 확률변수 생성
rv = stats.t(n)
xs = np.linspace(-3, 3, 100)

# y값을 직접 계산해서 생성한 t_sample의 히스토그램
ax.hist(t_sample, bins=100, range=(-3, 3), density = True, alpha=0.5, label='t_sample')

# 확률밀도함수(pdf) 이용하여 xs에 대응하는 y값을 계산하고 그래프 그리기
ax.plot(xs, rv.pdf(xs), label=f't({n})', color='gray')

ax.legend()
ax.set_xlim(-3, 3)
plt.show()


# In[60]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

xs = np.linspace(-3, 3, 100)
for n, ls in zip([3, 5, 10], linestyles):
    rv = stats.t(n)
    ax.plot(xs, rv.pdf(xs),
            label=f't({n})', ls=ls, color='gray')
rv = stats.norm()
ax.plot(xs, rv.pdf(xs), label='N(0, 1)')
    
ax.legend()
plt.show()


# In[61]:


rv = stats.t(5)
rv.isf(0.05)


# ![](./t분포정리.png)

# ## F 분포 - 분산분석 등에서 사용되는 확률분포
# ![](./f분포.png)
# 
# - Y1과 Y2 확률변수 모두 카이제곱분포를 따름
#     - 카이제곱분포의 특징 : 왼쪽으로 치우친 모양
#     

# In[66]:


n1 = 5
n2 = 10

rv1 = stats.chi2(n1)
rv2 = stats.chi2(n2)

sample_size = int(1e6)

Y1 = rv1.rvs(sample_size)
Y2 = rv2.rvs(sample_size)

f_sample = (Y1/n1)/(Y2/n2) # F 분포를 따르는 sample


# In[69]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# 자유도가 n1, n2 인  F 분포를 따르는 확률변수 rv를 생성
rv = stats.f(n1,n2)
xs = np.linspace(0,6,200) # 선 그래프

ax.plot(xs, rv.pdf(xs),label=f'F({n1}, {n2})', color='gray')

ax.hist(f_sample, bins=100, range=(0, 6),
        density=True, alpha=0.5, label='f_sample')

ax.legend()
ax.set_xlim(0, 6)
plt.show()


# In[70]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

xs = np.linspace(0, 6, 200)[1:]
for n1, ls in zip([3, 5, 10], linestyles):
    rv = stats.f(n1, 10)
    ax.plot(xs, rv.pdf(xs),
            label=f'F({n1}, 10)', ls=ls, color='gray')
    
ax.legend()
plt.show()

# 자유도가 낮을수록 왼쪽에 더 치우친 결과가 나타나게 됨

