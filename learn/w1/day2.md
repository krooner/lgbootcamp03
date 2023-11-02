# Day 2

## 2. ML Basic

### ML의 기본 프로세스
1. Set Hypothesis: ML은 학습하고자 하는 가설 (모델) 을 수학적 표현식으로 나타냄
    - 일련의 현상 (상황) 을 설명하기 위해 설정된 가정 (논리적 명제, 수식)
2. Set Cost function (Loss function)
    - 설정한 가설을 통한 예측값과 실제값의 오차를 평가하는 함수
3. Set Learning algorithm and Train a model
    - 학습 알고리즘의 목적은 Cost의 최소화
    - Hypothesis가 문제를 해결하기 위한 모델과 유사하도록 학습 진행


### 1) Linear Regression
Data를 잘 표현하는 직선의 방정식을 찾는 것
1. Hypothesis: Data를 대표하는 Hypothesis를 직선의 방정식으로 설정
    - $y=Ax+B$ (A: gradient, B: bias)
2. Cost function
    - Error 정의 (입력에 대한 Hypothesis의 결과와 Label의 차이)
    - 평균 오차 (Average Error)? 제곱의 평균 오차 (Mean Squared Error)?
3. Learning algorithm
    - Cost가 최소가 되도록 Hypothesis의 parameter를 조정하는 것.

#### Gradient descent
경사하강법. 선의 경사를 따라가며 최저점을 탐색하는 알고리즘
1. 임의의 Parameter 값에서 시작
2. Parameter를 조정 (Cost가 줄어들 수 있는 방향으로 변경)
    - Cost function의 기울기를 기준으로 조정
    - 기울기는 미분을 통해 얻어냄 (Parameter별 편미분)
3. Cost가 최저점에 도달할 때까지 반복

#### Learning rate
학습율. 성능 향상을 위한 Parameter update 시 변화의 정도를 조절하는 상수
- Cost function의 기울기를 기준으로 Parameter를 조정
    - Learning rate이 너무 크다면? Cost가 발산하게 될 수 있음
    - Learning rate이 너무 작다면? 학습 과정이 너무 오래 걸림
- 적절히 작은 값을 Learning rate으로 선택하는 것이 중요하다.

#### Convex function and Local/Global Minima
1. Convex function의 경우 `Local minimum == Global minimum`
    - Gradient descent 방식으로 Global minimum에 도달할 수 있음
2. Non-convex function의 경우 `Local minimum != Global minimum`
    - Gradient descent 방식이 적절할까?

```python
x_input = np.array(range(10), dtype=np.float32)
labels = np.array([2*i+1 for i in range(1, 11)], dtype=np.float32)

W, B = np.random.normal(), np.random.normal()

def Hypothesis(x): 
    return W*x+B
def Cost(): 
    return np.mean((Hypothesis(x_input)-labels)**2)
def Gradient_1(x, y): 
    # Partial derivative of W and B respectively
    return np.mean(x*(x*W+(B-y))), np.mean((W*x-y+B))

def Gradient_2(x, y):
    global W, B
    pres_w, pres_b = W, B
    delta = 5e-7 # sufficiently small value

    W = pres_w + delta
    cost_p = Cost()
    W = pres_w - delta
    cost_m = Cost()

    grad_w = (cost_p - cost_m)/(2*delta)
    W = pres_w

    B = pres_b + delta
    cost_p = Cost()
    B = pres_b - delta
    cost_m = Cost()

    grad_b = (cost_p - cost_m)/(2*delta)
    B = pres_b

    return grad_w, grad_b

epochs=5000
learning_rate=.005
for cnt in range(epochs+1):
    if cnt % (epochs//20) == 0:
        print("[{}]cost={},W={},B={}".format(cnt, Cost(), W, B))

    grad_w, grad_b = Gradient_1(x_input, labels)
    W -= learning_rate * grad_w
    B -= learning_rate * grad_b
```

#### Example: Blood Pressure Prediction Model
1. Set hypothesis
2. Set cost function
3. Train

**NumPy-based implementation**

```python
# Input data
x_input = tf.constant([25, 25, 25, 35, ..., 73], dtype=tf.float32)
labels = tf.constant([118, 125, 130, ... 138], dtype=tf.float32)

# Set Hypothesis, Cost function, and Gradient descent algorithm
def Hypothesis(x): 
    return W*x+B
def Cost(): 
    return np.mean((Hypothesis(x_input)-labels)**2)
def Gradient(x, y): 
    # Partial derivative of W and B respectively
    return np.mean(x*(x*W+(B-y))), np.mean((W*x-y+B))

# Training
epochs=100
learning_rate=.01
for cnt in range(epochs+1):
    if cnt % (epochs//20) == 0:
        print("[{}]cost={},W={},B={}".format(cnt, Cost(), W, B))

    grad_w, grad_b = Gradient(x_input, labels)
    W -= learning_rate * grad_w
    B -= learning_rate * grad_b

# Prediction
def Predict(x): return Hypothesis(x)
age = np.array([50.])
print("{} Years : {:>7.4}mmHg".format(age[0], Predict(age)[0]))
```

**TF-based implementation**
1. Set hypothesis and cost function (H(x) and Cost())
2. Set optimizer (모델을 학습하기 위해 사용하는 Class)
    - SGD (Stochastic Gradient Descent), Adam, ...
3. `optimizer.minimize(loss_function, var_list)`
    - loss_function는 arguments가 없어야 하며, var_list는 최적화할 변수 목록을 리스트나 튜플로 전달 `[W, B]` 
    - 내부적으로 `tf.GradientTape`과 `apply_gradient()`를 사용한다.
4. Predict

```python
import numpy as np
import tensorflow as tf

print("NumPy version: {}".format(np.__version__))
print("TensorFlow version: {}".format(tf.__version__))

# Input data
x_input = tf.constant([25, 25, 25, 35, ..., 73], dtype=tf.float32)
labels = tf.constant([118, 125, 130, ... 138], dtype=tf.float32)

## Parameters
W = tf.Variable(tf.random.normal(()), dtype=tf.float32)
B = tf.Variable(tf.random.normal(()), dtype=tf.float32)

def Hypothesis(x): return x*W+B
def Cost(): return tf.reduce_mean(tf.square(Hypothesis(x_input)-labels))

# Training
epochs = 150000
learning_rate=0.0003
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

for cnt in range(1, epochs+1):
    if cnt % (epochs//20) == 0:
        print("[{}]cost={},W={},B={}".format(cnt, Cost(), W.numpy(), B.numpy()))

    optimizer.minimize(Cost, [W, B])

# Prediction
def Predict(x): return Hypothesis(x)
age = tf.constant([50.])
print("{} Years : {:>7.4}mmHg".format(age[0], Predict(age)[0]))
```
**NumPy vs TF-based implementation**

|Category|NumPy|TensorFlow|
|---|---|---|
|변수|numpy.ndarray|tf.Variable() and tf.Constant()|
|구현|np.mean()과 같은 method 사용|tf.reduce_mean(), tf.square()와 같은 method 사용|
|학습|직접 구현|제공되는 Optimizer.minimize 사용|

### 2) Multi-variable Linear Regression
입력이 여러 개 $x_{1}, x_{2}, ..., x_{i}$

**NumPy implementation**

```python
x_input, labels = np.array(...), np.array(...)

W, B = np.random.normal(size=(2, 1)), np.random.normal(size=())

def Hypothesis(X): return np.matmul(X, W) + B
def Cost(): return np.mean((Hypothesis(x_input) - labels)**2)
def Gradient():
    global W, B
    pres_W, grad_W = W.copy(), np.zeros_like(W)
    delta = 5e-7

    for idx in range(W.size):
        W[idx, 0] = pres_W[idx, 0] + delta
        cost_p = Cost()

        W[idx, 0] = pres_W[idx, 0] - delta
        cost_m = Cost()

        grad_W[idx, 0] = (cost_p - cost_m)/(2*delta)
        w[idx, 0] = pres_W[idx, 0]

    pres_b = B
    B = pres_b + delta
    cost_p = Cost()

    B = pres_b - delta
    cost_m = Cost()
    grad_B = (cost_p-cost_m)/(2*delta)
    B = pres_b

    return grad_W, grad_B

epochs = 1e6
learning_rate = 1e-4

training_idx = np.arange(0, epochs+1, 1)
cost_graph = np.zeros(epochs+1)

for cnt in range(0, epochs+1):
    cost_graph[cnt] = Cost()
    if cnt % (epochs//20) == 0:
        print("[{}]cost={},W=[[{}][{}]],B={}".format(cnt, cost_graph[cnt], W[0,0], W[1,0], B))

    grad_W, grad_B = Gradient()
    W -= learning_rate * grad_W
    B -= learning_rate * grad_B

# Plot
plt.title("Cost/Epochs graph")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.plot(training_idx, cost_graph)
plt.xlim(0, epochs)
plt.grid(True)
plt.semilogy() # set y-axis scale to log-scale
plt.show()
```

**TF Implementation (with Data Normalization)**
Data Normalization은 Cost 값의 발산으로 Learning rate을 낮게 설정해야 하는 문제를 개선

```python
x_input_org = x_input
x_min, x_max = np.min(x_input, axis=0), np.max(x_input, axis=0)
x_input = (x_input-x_min)/(x_max-x_min)
...

# Prediction
## input_value: [0, 1]
## output_value: original_value
def predict(x): return Hypothesis((x-x_min)/(x_max-x_min))
```

### 3) Multi-variable Multi-output Linear Regression
`여러 개의 입력`에 따른 `여러 개의 출력`을 예측할 때 사용 가능
- 나이와 BMI가 입력하면, 수축기 혈압과 이완기 혈압을 예측
- $f(x_1, x_2, ..., x_n) = y_1, y_2, ..., y_m$
    - input_sample_num $N$
    - sample_input_dim $n$
    - sample_output_dim $m$

$X\in R^{N\times n}, W\in R^{n\times m}, b\in R^{m}, Y\in R^{N\times m}$

```python
W = tf.Variable(tf.random.normal((2, 2)), dtype=tf.float32)
B = tf.Variable(tf.random.normal((2,)), dtype=tf.float32)
```

## Appendix: TensorFlow
Google이 2015년 11월에 공개한 End-to-end 오픈소스 머신러닝 플랫폼. Tensor를 포함한 연산을 정의하고 실행하는 프레임워크.
- Model Build가 쉽다: 다양한 수준의 추상화 제공.
- 서버, 엣지 디바이스, 웹에서 언어/플랫폼 별 머신러닝 모델을 쉽게 학습 및 배포 가능 (TensorFlow Lite)

Tensor
- 수학/과학: `A tensor is a geometric object that maps in a multi-linear manner geometric vectors, scalars, and other tensors to a resulting tensor`
- TensorFlow: `Tensors are multi-dimensional arrays with a uniform type`

### Tensor의 기본 속성
- dtype (data type): Tensor 값의 Type이며, Tensor 값의 각 Element들은 모두 같은 dtype을 가짐.
- shape: 각 차원에 대한 크기를 가지고 있는 숫자들
    - `?`인 경우, 해당 크기는 Runtime 시 상황에 따라 달라질 수 있음을 의미
- Tensor 생성과 관련된 클래스 및 메서드

|Name|Format|Description|
|---|---|---|
|tf.constant|Method|Constant tensor를 생성|
|tf.Variable|Class|Variable tensor와 관련|

### Constant tensor (tf.constant)

`tf.constant(value, dtype=None, shape=None, name='Const')`
- Attributes
    - value (A constant value or list of output dtype)
    - dtype (The type of the elements of the resulting tensor)
    - name (Optional name for the tensor)
- Rank of value

|Rank|Description|Example|Shape|
|---|---|---|---|
|0|Scalar (Magnitude)|1, -4, .142|()|
|1|Vector (Magnitude and direction)|[1., 2., 3.]|(3, )|
|2|Matrix (Table of number)|[[1., 2., 3.], [4., 5., 6.]]|(2, 3)|
|3|3-Tensor (Cube)|[[[1., 2., 3.]], [[4., 5., 6.]]]|(2, 1, 3)|

```python
import tensorflow as tf

c_int = tf.constant(3)
c_float1 = tf.constant(1.239)
c_float2 = tf.constant(3, dtype=tf.float32)
c_str = tf.constant("Hello, TensorFlow", name="Const_String")
list_constant = [c_int, c_float1, c_float2, c_str]

for tensor in list_constant:
    print(tensor, "value={}".format(tensor.numpy()))

```

|tf.dtype|Description|
|---|---|
|tf.float{16, 32, 64}|{half, single, double}-precision floating-point|
|tf.bfloat16|16-bit truncated floating-point|
|tf.int{8, 16, 32, 64}|signed integer|
|tf.uint{8, 16, 32, 64}|unsigned integer|
|tf.bool|Boolean|
|tf.string|String|

Tensor with higher Ranks
- Batch operation에서 Tensor의 first dimension은 batch_dim
- last dimension은 num_of_channel
- 나머지 dimension은 space dimension
- `(batch_size, height, width, channel)`

### tf.Variable
Operation 실행에 의해 값이 조정되는 Tensor
`tf.Variable(initial_value=None, trainable=True, dtype=None, ...)`

|Name|Description|
|---|---|
|initial_value|Variable의 초깃값|
|trainable|True일 경우, Optimizer 사용 시 해당 변수를 Training|
|dtype|Variable의 Type|

**Variable의 초깃값 설정에 사용할 수 있는 함수**

|Name|Description|
|---|---|
|tf.zeros(shape, dtype)|Shaped tensor with zero-values|
|tf.ones(shape, dtype|Shaped tensor with one-value|
|tf.eye(num_rows, num_columns, dtype)|identity matrix|
|tf.random.normal(shape, mean, stddev, dtype, seed, name)|Following Normal distribution|
|tf.random.truncated_normal(shape, mean, stddev, dtype, seed, name)|Following Truncated normal distribution|
|tf.random.uniform(shape, minval, maxval, dtype, seed, name)|Following uniform distribution|