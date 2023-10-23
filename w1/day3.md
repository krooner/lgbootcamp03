# Day 3

## 2. ML Basic

### Logistic Regression for Binary Classification
- Linear Regression: 주어진 입력에 따라 Continuous한 값을 예측 (for Regression)
- Logistic Regression: 주어진 입력에 따라 Discrete한 값을 예측 (for Classification)
    - Hypothesis로 Logistic function (Curve) 를 적용하여 Regression
    - Sigmoid function (Standard logistic function) `(0, 1)`

#### Hypothesis
데이터를 가장 잘 표현할 수 있는 W, b 값을 학습
- $H(x) = \frac{1}{1+\exp{(-(WX+b))}}$

**Decision boundary**: Sigmoid function의 결과를 0, 1 둘 중 하나로 판정하는 기준이자 경계

#### Probability, Odds, Logit, and Sigmoid
- Probability (확률) $p$
- Odds (승산) $\frac{p}{1-p}$
- Logit (Logistic + probit): 로그를 사용해서 확률을 측정
    - probit (probability + unit): 확률을 재는 단위
    - $\log{\frac{p}{1-p}}$
- Sigmoid
    - $p = \frac{1}{1+\exp{(-\log{\frac{p}{1-p}})}}$

ML에서 Logit이라 하면 $WX+b$를 지칭한다.

#### Cost function: Cross-Entropy Loss
- Cost function이 Non-convex 형태이므로, Gradient descent 방식으로 최적화하기 어려움
- Sigmoid 함수를 Hypothesis로 사용하는 경우, 다른 형태의 Cost function을 필요로 함
    - **Cross-Entropy Error** 두 확률분포 $p, q$의 차이
        - $CEE = -\sum_{i=1}^{m}p_{i}\log{q_i}$ 
    - Entropy (시스템 내의 무질서도를 측정하는 척도)

#### Google Colaboratory Practices

### Softmax Classification for Multinomial Classification
Classify data with more than 3 classes?
1. Multiple 1 vs 1 models
2. Single model (with Softmax function)

#### Softmax function
- logit $\frac{y_i}{\sum_{j=0}^{m-1}y_j}$ 대신 odds $\frac{e^{y_i}}{\sum_{j=0}^{m-1}e^{y_j}}$
- 확률 차이가 크면 Label과 Hypothesis 간 차이가 커져서 학습 효율이 향상됨
- Logit에 Sigmoid을 취해서 Probability를 구한 뒤 Odds를 구하는 대신, 
    - Logit에 바로 Exponential을 취함 

#### Cost function for Softmax Classification
- Label은 One-hot Encoding
- Cross Entropy Error $-\sum_{i=0}^{m-1}{p_{i}\log{q_i}}$
    - Hypothesis의 결과가 사건의 확률이며 Label과의 차이를 계산

**NumPy implementation**

```python

x_input = ...
labels = ...

n_var, n_class = 2, 3
W = np.random.normal(size=(n_var, n_class))
B = np.random.normal(size=(n_class, ))

def Softmax(y):
    c = np.max(y, axis=-1).reshape((-1, 1))
    exp_y = np.exp(y-c)

    sum_exp_y = np.sum(exp_y, axis=1).reshape((-1, 1))
    res_y = exp_y / sum_exp_y
    return res_y

def Hypothesis(x): return Softmax(np.matmul(X, W)+B)
def Cost():
    return np.mean(-np.sum(labels * np.log(Hypothesis(x_input)), axis=1))

```

**TF-based implementation**