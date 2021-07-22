#!/usr/bin/env python
# coding: utf-8

# # 연속형 확률변수

# ## 연속형 확률변수
# 
# - 확률변수가 취할 수 있는 값이 연속적인 확률변수
# - 특정 값을 취하는 확률은 정의되지 않음
# - 확률변수가 어느 구간에 들어가는가에 대한 확률을 정의
# 
# 
# - [예] 룰렛 :
#     - 취할 수 있는 값이 0부터 1 사이의 실수
#     - 큰 수일수록 나오기 쉬워지는 불공정한 구조
#     - 0.5 라는 값을 취할 확률 = 0
#         - 정확하게 0.5000000000000... 를 취할 가능성이 없으므로 확률은 0

# ### 확률 밀도 함수
# 
# - 확률변수가 취할 수 있는 값은 구간[a, b]으로 결정됨
# - 확률 밀도 함수(PDF) 또는 밀도함수 f(x)로 정의
# 
# 
# - 어떤 특정 값을 취하는 확률로는 정의되지 않음
#     - 𝑓(𝑥) ≠𝑃(𝑋=𝑥)
# 
# ![](./밀도적분.png)
# 
# 
# - 이 적분은 밀도함수 f(x)와 x축, 그리고 두 직선 x=x0, x=x1으로 둘러싸인 영역의 면적으로 해석할 수 있음
#     - 그림에서 칠해진 면적이 확률 P

# #### 불공정한 룰렛을 예로 들어 코드를 구현
# - 취할 수 있는 값이 0부터 1 사이의 실수
# - 큰 수일수록 나오기 쉬워지는 불공정한 구조

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


from scipy import integrate
import warnings

# 적분에 관한 warning을 출력하지 않도록 한다
warnings.filterwarnings('ignore',
                        category=integrate.IntegrationWarning)


# In[3]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


# In[4]:


# 취할 수 있는 값의 구간을 정의
x_range = np.array([0,1])


# - x_range를 정의역으로 하는 밀도 함수를 구현
#     - 불공정한 룰렛은 큰 값일수록 나오기 쉬움 → 아래와 같은 밀도함수로 정의
# ![](./룰렛밀도함수.png)

# In[5]:


def f(x) :
    if x_range[0] <= x <= x_range[1] :
        return 2 * x
    else :
        return 0


# In[6]:


f(0.4)
# 이 식은 0.4가 나올 확률이 아님
# 연속형이므로 두 구간 경계 위치값을 함수를 통해 구해서 면적을 그린 후 해당 면적을 적분하여 구함
# 반환되는 면적의 값이 두 구간 범위의 확률임


# In[7]:


# 확률변수 X를 정의
x = [x_range, f]


# - 위에서 작성한 밀도함수 f(x)를 그래프로 그림
#     - 확률의 이미지를 쉽게 전달하기 위해 f(x)와 x축, 두 직선 x=0.4, x=0.6의 영역에 색을 적용
#     - 불공정한 룰렛이 0.4부터 0.6 사이의 값을 취할 확률

# In[10]:


xs = np.linspace(x_range[0], x_range[1], 100) # x=0, x=1 사이에서 발생할 수 있는 100개의 값을 추출 = x축의 값 

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# 확률함수 그래프
ax.plot(xs, [f(x) for x in xs], label='f(x)', color='gray')
# y 값 = [f(x) for x in xs]

#ax.hlines(y, xmin, xmax, alpha=)
ax.hlines(0, -0.2, 1.2, alpha=0.3)

#ax.vlines(x, ymin, ymax, alpha=)
ax.vlines(0, -0.2, 2.2, alpha=0.3)
ax.vlines(xs.max(), 0, 2.2, linestyles=':', color='gray')

# 0.4 부터 0.6 까지 x 좌표를 준비
xs_p = np.linspace(0.4, 0.6, 100)

# xs_p의 범위로 f(x)와 x축으로 둘러싸인 영역을 진하게 칠함
ax.fill_between(xs_p, [f(x) for x in xs_p], label='prob')

ax.set_xticks(np.arange(-0.2, 1.3, 0.1))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.2, 2.1)
ax.legend()

plt.show()


# ### plt.plt.fill_between() 예시

# In[11]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

plt.plot(x, y) # 원 소스의 그래프를 그리고, 그 중에 알고 싶은 확률만 면적을 색칠해서 표시
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.fill_between(x[1:3], y[1:3], alpha=0.5) ## fill_between() 사용
x[1:3], y[1:3]
# 네 점 (x[1], y[1]), (x[2], y[2]), (x[1], 0), (x[2], 0)을 잇는 영역이 채워짐

plt.show()


# ### 연속형 확률변수의 확률의 성질
# 
# ![](./확률성질.png)

# - 첫 번째 성질 : 함수 f(x)는 0보다 같거나 커야 함
#     - 위 성질을 확인하기 위해서는 minimize_scalar 함수 사용
#     - minimize_scalar() : 함수를 실행한 결과값이 최소가 되는 x값과 최소 결과 y를 반환

# In[14]:


from scipy.optimize import minimize_scalar

def func(x) :
    return (x-1.5)**2 + 0.5

res = minimize_scalar(func)
res.x # 최소값을 만들어 낸 x값
res.fun # 함수가 만들어낼 수 있는 최소값


# In[15]:


# 위에서 생성해 놓은 불공정 룰렛의 확률함수
def f(x):
    if x_range[0] <= x <= x_range[1]:
        return 2 * x
    else:
        return 0


# In[18]:


from scipy.optimize import minimize_scalar

res = minimize_scalar(f)
res.x
res.fun # 0이 반환됐으므로 첫 번째 확률의 성질을 만족


# - 위 연산의 확인으로 인해 f(x) >= 0 이라는 연속형 확률변수의 성질이 만족함을 알 수 있음

# #### 연속형 확률변수의 두 번째 확률의 성질
# 
# ![](./확률성질2.png)

# - 두번째 f(x)를 －∞ 부터 ∞ 까지 적분한 결과가 1이라는 것은 위 그림에서 삼각형의 면적이 1이 되는 것과 같다
#     - 이 삼각형은 밑변의 길이가 1, 높이가 2 이므로 면적이 1이라는 것을 간단히 알 수 있음

# ### 확률의 두 번째 성질을 수치적분 함수를 사용해 만족하는지 확인
# 
# #### quad() : 수치적분 함수
# 
# - **수치적분(numerical integration)**은 함수를 아주 작은 구간으로 나누어 실제 면적을 계산함으로써 정적분의 값을 구하는 방법이다. 
# - Scipy의 integrate 서브패키지의 quad 명령으로 수치적분을 할 수 있다.
# 
# - 첫 번째 인수는 피적분함수、두 번째 인수와 세 번째 인수는 적분 범위
#     - __첫 번째 반환값이 수치 적분으로 얻어진 결과(면적)이며, 두 번째의 값은 추정 오차__
# 

# In[19]:


# 두 번째 성질을 적분 계산으로 확인
# -무한대와 +무한대는 np.inf로 표현할 수 있음 -> np.inf : 무한대 | -np.inf : 마이너스무한대
integrate.quad(f, -np.inf, np.inf)

# f(x)를 -무한대부터 +무한대까지 적분한 결과가 1이어야 함 - 두번째 확률의 성질


# In[21]:


xs = np.linspace(x_range[0], x_range[1], 100) # x=0, x=1 사이에서 발생할 수 있는 100개의 값을 추출 = x축의 값 

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# 확률함수 그래프
ax.plot(xs, [f(x) for x in xs], label='f(x)', color='gray')
# y 값 = [f(x) for x in xs]

#ax.hlines(y, xmin, xmax, alpha=)
ax.hlines(0, -0.2, 1.2, alpha=0.3)

#ax.vlines(x, ymin, ymax, alpha=)
ax.vlines(0, -0.2, 2.2, alpha=0.3)
ax.vlines(xs.max(), 0, 2.2, linestyles=':', color='gray')

# 0.4 부터 0.6 까지 x 좌표를 준비
xs_p = np.linspace(0.4, 0.6, 100)

# xs_p의 범위로 f(x)와 x축으로 둘러싸인 영역을 진하게 칠함
ax.fill_between(xs_p, [f(x) for x in xs_p], label='prob')

ax.set_xticks(np.arange(-0.2, 1.3, 0.1))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.2, 2.1)
ax.legend()

plt.show()


# - 위 사다리꼴 영역을 적분으로 구하는 식
# ![](./적분식.png)

# In[24]:


# 2x는 함수 f(x)가 구현하고 있음
integrate.quad(f, 0.4, 0.6)

# 적분값이 0.200 이므로 0.4에서 0.6이 나올 확률은 0.2


# ### 누적 분포 함수
# 
# - 확률변수 X가 x 이하가 될 때의 확률을 반환하는 함수
# ![](./누적분.png)

# In[25]:


# 분포함수 구현

def F(x) :
    return integrate.quad(f, -np.inf, x)[0] # [0]이 적분값에 해당하고, [1]은 추정 오차값에 해당함


# - 룰렛 0.4에서 0.6 사이의 값을 취할 확률
# ![](./누적분2.png)

# In[26]:


F(0.6) - F(0.4)


# In[27]:


# 단조증가함수 확인

xs = np.linspace(x_range[0], x_range[1], 100)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

ax.plot(xs, [F(x) for x in xs], label='F(x)', color='gray')
ax.hlines(0, -0.1, 1.1, alpha=0.3)
ax.vlines(0, -0.1, 1.1, alpha=0.3)
ax.vlines(xs.max(), 0, 1, linestyles=':', color='gray')

ax.set_xticks(np.arange(-0.1, 1.2, 0.1))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.legend()

plt.show()


# ### 연속형 확률 분포의 지표
# 
# #### 평균 - 기대값
# ![](./연속기대.png)

# In[28]:


# 위에서 생성해 놓은 불공정 룰렛의 확률함수
def f(x):
    if x_range[0] <= x <= x_range[1]:
        return 2 * x
    else:
        return 0


# In[29]:


def integrand(x) : # (x*f(x)) 계산 결과를 반환하는 함수 = 피적분함수에 해당
    return x * f(x)

integrate.quad(integrand, -np.inf, np.inf)[0]


# In[30]:


# 확률변수 X를 정의
X = [x_range, f]


# In[33]:


def E(X, g=lambda x: x) :
    x_range, f = X
    def integrand(x) : # (x(f(x))) 계산 결과를 반환하는 함수
        return g(x) * f(x)
    return integrate.quad(integrand, -np.inf, np.inf)[0]

# sub함수는 E() 함수가 호출하는 순간 생성됨    


# In[34]:


E(X)


# In[35]:


E(X, g=lambda x : 2*x+3)


# In[36]:


2 * E(X) + 3


# ### 분산
# 
# - μ 는 확률변수 X의 기대값
# 
# ![](./분산.png)

# In[38]:


# 분산 계산
mean = E(X) # 확률변수 X에 대한 기대값

def integrand(x) :
    return (x-mean)**2 * f(x)

integrate.quad(integrand, -np.inf, np.inf)[0]


# ![](./분산식.png)

# - 위 수식을 함수로 구현

# In[41]:


def V(X, g = lambda x : x) :
    x_range, f = X
    mean = E(X,g) # 기대값 계산
    
    def integrand(x) :
        return (g(x)-mean) **2 * f(x)
    
    return integrate.quad(integrand, -np.inf, np.inf)[0]


# In[42]:


V(X)


# In[43]:


# 확률변환변수의 분산
V(X, lambda x: 2*x+3)


# In[44]:


# 이산형 확률 변수에서 확인했던 분산의 성질은 연속형 확률변수에도 적용
# 성질에 따라 아래 수식으로도 계산 가능
2**2*V(X)

