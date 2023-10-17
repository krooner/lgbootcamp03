# Day 2

## ML Basic

ML의 기본 프로세스
1. Set Hypothesis: ML은 학습하고자 하는 가설 (모델) 을 수학적 표현식으로 나타냄
    - 일련의 현상 (상황) 을 설명하기 위해 설정된 가정 (논리적 명제, 수식)
2. Set Cost function (Loss function)
    - 설정한 가설을 통한 예측값과 실제값의 오차를 평가하는 함수
3. Set Learning algorithm and Train a model
    - 학습 알고리즘의 목적은 Cost의 최소화
    - Hypothesis가 문제를 해결하기 위한 모델과 유사하도록 학습 진행


### Linear Regression
Data를 잘 표현하는 직선의 방정식을 찾는 것
1. Hypothesis: Data를 대표하는 Hypothesis를 직선의 방정식으로 설정
    - $y=Ax+B$ (A: gradient, B: bias)
2. Cost function
    - Error 정의 (입력에 대한 Hypothesis의 결과와 Label의 차이)
    - 평균 오차 (Average Error)? 제곱의 평균 오차 (Mean Squared Error)?
3. Learning algorithm
    - Cost가 최소가 되도록 Hypothesis의 parameter를 조정하는 것.

## Gradient descent: 경사하강법

곡선의 경사를 따라가며 최저점을 탐색하는 알고리즘
1. 임의의 Parameter 값에서 시작
2. Parameter를 조정 (Cost가 줄어들 수 있는 방향으로 변경)
    - Cost function의 기울기를 기준으로 조정
    - 기울기는 미분을 통해 얻어냄 (Parameter별 편미분)
3. Cost가 최저점에 도달할 때까지 반복

### Learning rate: 학습율
성능 향상을 위한 Parameter update 시 변화의 정도를 조절하는 상수
- Cost function의 기울기를 기준으로 Parameter를 조정
    - Learning rate이 너무 크다면? Cost가 발산하게 될 수 있음
    - Learning rate이 너무 작다면? 학습 과정이 너무 오래 걸림
- 적절히 작은 값을 Learning rate으로 선택하는 것이 중요하다.

### Convex function and Local/Global Minima
1. Convex function의 경우 Local minimum == Global minimum
    - Gradient descent 방식으로 Global minimum에 도달할 수 있음
2. Non-convex function의 경우 Local minimum != Global minimum
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

## Example: Blood Pressure Prediction Model
1. Set hypothesis
2. Set cost function
3. Train

### Naive implementation

```python
x_input = tf.constant([25, 25, 25, 35, ..., 73], dtype=tf.float32)
labels = tf.constant([118, 125, 130, ... 138], dtype=tf.float32)

def Hypothesis(x): 
    return W*x+B
def Cost(): 
    return np.mean((Hypothesis(x_input)-labels)**2)
def Gradient(x, y): 
    # Partial derivative of W and B respectively
    return np.mean(x*(x*W+(B-y))), np.mean((W*x-y+B))

epochs=100
learning_rate=.01
for cnt in range(epochs+1):
    if cnt % (epochs//20) == 0:
        print("[{}]cost={},W={},B={}".format(cnt, Cost(), W, B))

    grad_w, grad_b = Gradient(x_input, labels)
    W -= learning_rate * grad_w
    B -= learning_rate * grad_b

def Predict(x): return Hypothesis(x)

age = np.array([50.])
print("{} Years : {:>7.4}mmHg".format(age[0], Predict(age)[0]))
```


### TF-based implementation
1. Set hypothesis and cost function (H(x) and Cost())
2. Set optimizer (모델을 학습하기 위해 사용하는 Class)
3. `optimizer.minimize(loss, var_list)`
    - loss는 arguments가 없어야 하며, var_list는 최적화할 변수 목록을 리스트나 튜플로 전달 
    - 내부적으로 `tf.GradientTape`과 `apply_gradient()`를 사용한다.
4. Predict

```python
import numpy as np
import tensorflow as tf

print("NumPy version: {}".format(np.__version__))
print("TensorFlow version: {}".format(tf.__version__))

x_input = tf.constant([25, 25, 25, 35, ..., 73], dtype=tf.float32)
labels = tf.constant([118, 125, 130, ... 138], dtype=tf.float32)

# Parameters
W = tf.Variable(tf.random.normal(()), dtype=tf.float32)
B = tf.Variable(tf.random.normal(()), dtype=tf.float32)

def Hypothesis(x): # Hypothesis
    return x*W+B
def Cost(): # Cost (Loss) function
    return tf.reduce_mean(tf.square(Hypothesis(x_input)-labels))

# Hyper-parameters
epochs = 150000
learning_rate=0.0003
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

for cnt in range(1, epochs+1):
    if cnt % (epochs//20) == 0:
        print("[{}]cost={},W={},B={}".format(cnt, Cost(), W.numpy(), B.numpy()))

    optimizer.minimize(Cost, [W, B])

def Predict(x):
    return Hypothesis(x)

age = tf.constant([50.])
print("{} Years : {:>7.4}mmHg".format(age[0], Predict(age)[0]))

```

## NumPy 코드와 TensorFlow 코드

|Category|NumPy|TensorFlow|
|---|---|---|
|변수|numpy.ndarray|tf.Variable() and tf.Constant()|
|구현|np.mean()과 같은 method 사용|tf.reduce_mean(), tf.square()와 같은 method 사용|
|학습|직접 구현|제공되는 Optimizer.minimize 사용|

## Multi-variable Linear Regression

``````

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
|0|Scalar|1, -4, .142|()|
|1|Vector|[1., 2., 3.]|(3, )|
|2|Matrix|[[1., 2., 3.], [4., 5., 6.]]|(2, 3)|
|3|3-Tensor|[[[1., 2., 3.]], [[4., 5., 6.]]]|(2, 1, 3)|

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

### tf.dtype
Tensor의 요소 타입
|Type|Description|
|---|---|
|tf.float{16, 32, 64}||
|tf.bfloat16||
|tf.
||
|||
|||
|||