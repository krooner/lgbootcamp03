# Day 1

## Introduction to ML/DL

### AI
- AI란
    - `기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것` (John McCarthy Dartmouth, 1956)
- AI 기술에 대한 접근방식의 변화
    - Logic and Rule-based: 전문가가 정립한 논리와 규칙에 따라 기계가 판단
    - Machine Learning (ML): 전문가들이 모델링한 규칙을 데이터를 통해 기계가 학습
    - Deep Learning (DL): 모델링한 규칙 없이 스스로 학습
- AI 발전방향
    - ML: 데이터를 이용해서 컴퓨터가 어떤 지식이나 패턴 학습
    - DL: 다층구조 형태의 신경망 기반 ML
- AI 분류 (By McKinsey & Company)
    - ML, Robotics, ANN (Artificial Neural Networks)
- AI 활용
    - Autonomous Driving, Medical, Image Classification, Robotics, ...
- 딥러닝이 주목 받는 이유
    1. 빅데이터의 발달 (수집 가능한 데이터의 급증)
    2. 컴퓨팅 파워의 개선 (GPU 발달 및 클라우드 환경)
    3. 딥러닝 알고리즘의 향상 (CNN, RNN)
    4. 오픈소스 프레임워크 지원
- 딥러닝 적용 사례: AlphaGo
    - 바둑을 선택한 이유: 기존 컴퓨팅 파워로는 모든 경우의 수 파악 불가 (체스: $10^{47}$, 바둑: $10^{170}$)
    - 학습 방식 (다양한 방식으로 성능 개선)
        1. 3천만 건의 기보 데이터로 지도학습 및 스스로 경기를 하며 강화학습
        2. 두 종류의 신경망을 이용하도록 개발. 자체 신경망끼리 수백만회의 바둑 실전
            - Value Network (각 수에 대한 위치 및 승률 평가)
            - Policy Network (좋은 수를 찾아내 움직임을 선택)
    - AlphaGo Zero 성능 개선

### ML and Perceptron

#### 고전적 프로그래밍의 한계
Explicit Programming
- 문제 해결을 위한 규칙을 프로그래머가 직접 결정. 프로그램은 이에 기반하여 작성됨.
- 난이도가 높음 (스팸 필터링, 자율 주행 등)
    - 다양한 상황에 따라, 정해진 규칙으로만 해결할 수 없는 문제도 존재
- Machine Learning: 명시적인 프로그래밍 없이 데이터를 이용하여 컴퓨터가 어떤 지식이나 패턴을 학습하는 것 (Arthur Samuel, 1959)

#### 공학적 접근 방법

```
특정 작업 T에 대한 프로그램의 성능을 P로 측정했을 때, 경험 E로 인해 성능이 향상되었다면 이 프로그램은 작업 T와 성능 평가 P에 대해 경험 E로 학습한 것 (Tom Mitchell, 1997)
```

예시
- Task: 메일의 스팸 여부 분류
- Experience: 스팸 여부 판정 과정 경험
- Performance measure: 매일의 스팸 분류 정확도

#### 머신러닝의 유형
기계 관점에서는 학습을 진행하는 것 (Learning). 사람 관점에서는 기계를 학습시키는 것 (Training).
1. 지도학습 (w/ Label)
    - Regression (회귀): 특정 실수 값을 예측하는 문제
    - Classification (분류): Multi-class Classification
2. 비지도학습 (w/o Label): Clustering
3. 강화학습

#### 지도학습
Labeled Data를 이용하는 학습 알고리즘
- 입력에 대한 프로그램의 출력 결과를 Label과 비교하여 학습 가능
- 학습 과정 (예: MNIST Image Classification)
    1. 손으로 쓴 숫자 이미지 입력: 문제 해결 연산으로 Output 생성
    2. Label과 Output을 비교하여 필요 시 문제 해결 연산을 개선: 학습

#### 학습과 뉴런
ANN (Artificial Neural Network)
- 생물학적 신경망에서 아이디어를 얻음
    - 사람의 뇌 속에는 $10^{11}$ 개의 뉴런이 있고, 각 뉴런은 $10^{3}$개의 다른 뉴런과 연결되어 있음
    - 이 연결을 통해 각 뉴런은 서로 전기 신호를 주고 받으며 연산 수행
- 컴퓨터도 인간의 뇌처럼 대량의 병렬 처리 연산을 수행할 수 있도록 설계

Neuron
- 수상돌기를 통해 다른 뉴런으로부터 전기 신호 받음
- 축삭돌기는 수신한 전기 신호를 말단에서 다른 뉴런으로 전달
- 수상돌기로부터 받은 전기 신호가 역치를 넘지 못하면 중간에 사라지기도 함

Perceptron
- 생물학적 뉴런을 공학적인 구조로 변형한 것으로, 인공신경망을 공학적으로 구현
- `The perceptron: A probabilistic model for information storage and organization in the brain (Frank Rosenblatt, 1958)`

|Category|Description|
|---|---|
|Input|하나 혹은 그 이상의 Perceptron 입력|
|Activation Function|Step function, Sigmoid, ReLU 등 입력에 가중치를 곱한 값의 총 합을 입력받아 정해진 출력을 내보내는 함수|
|Output|Activation Function의 출력 결과|

#### Google Colaboratory를 활용한 실습

### Appendix: NumPy

06년에 출시된 Python 수치해석용 라이브러리로 ML/DL에 필요한 다양한 기능 제공
- NumPy의 ndarray (N-dimensional array)
    - `An array object represents a multidimensional, homogeneous array of fixed-size items`
    - 과학적 계산을 위해 하드웨어가 효율적으로 처리할 수 있는 구조
    - C언어의 배열과 유사한 특징을 가짐: 연속적 메모리 배치

|Python List|NumPy ndarray|
|---|---|
|다른 타입의 아이템을 가질 수 있음|동일 타입의 아이템을 가짐|
|비연속적인 메모리 구조|연속적인 메모리 구조|
|Vectorizing 연산 불가능|Vectorizing 연산 가능|


#### np.ndarray

```python
import numpy as np
arr = np.ones((10, 1))
```

|Attribute (변수 개념)|Description|
|---|---|
|arr.ndim|배열의 깊이 (=차원)|
|arr.shape|각 차원별 요소 개수 정보를 가진 배열의 구조 정보|
|arr.dtype|생성된 배열의 데이터 타입에 대한 정보|
|arr.itemsize|Item의 메모리 사용 바이트 수|
|arr.size|배열의 Item 갯수 (2행 3열인 경우 size는 6)|

|Method (함수 개념)|Description|
|---|---|
|arr.reshape()|배열을 주어진 Shape으로 변경하여 반환|
|arr.transpose()|Axis가 전치된 배열을 반환|
|arr.copy()|배열을 복사하여 반환|

```python
a = np.ones((3, 3)) # one-value array
b = np.zeros((2, 2)) # zero-value array
c = np.eye(3) # identity array

a = [1, 2, 3, 4]
a_like = np.ones_like(a) 
# [1 1 1 1]

b = [[1, 2, 3], [4, 5, 6]]
b_like = np.zeros_like(b)
# [[0 0 0]
#  [0 0 0]]

c = [[1, 2], [3, 4]]
c_like = np.empty_like(c)
# [[1 2] 
#  [3 4]]

```

#### datatype

|Datatype|Example|
|---|---|
|signed int|int8, int16, int32, int64|
|unsigned int|uint8, uint16, uint32, uint64|
|float|float16, float32, float64, float128|
|complex|complex64, complex128, complex256|
|boolean|bool|
|string|bytes_{byte-string}, str_{unicode-string}|
|object|object (Container 자료형들은 object로 취급함)|


#### Indexing and Slicing

|Indexing|Slicing|
|---|---|
|Element의 Index number를 사용함. 다차원인 경우 Comma로 차원 구분|Element의 인덱스 번호와 Colon을 함께 사용. `start:end:step`|

#### Dimension reduction
배열의 차원을 감소시키는 연산

`np.sum(), np.ndarray.sum()`

```python
x = np.array([1, 2, 3, 4])
print(np.sum(x)) # 10
print(x.sum())

x = np.array([[1, 1], [2, 2]])
print(x.sum(axis=0)) # Vertical 방향으로 Gather
print(x.sum(axis=1)) # Horizontal 방향으로 Gather
print(x.sum()) # 6
```

`min/max, argmin/argmax`

```python
x = np.array([1, 3, 2])

x.min() # min-value
x.max() # max-value

x.argmin() # index-of-min-value
x.argmax() # index-of-max-value

x = np.array([[9, 2, 10, 6, 1], [5, 7, 3, 4, 8]])
x.argmin(axis=0) # 1 0 1 1 0
x.argmax(axis=1) # 2 4

```

`np.all(), np.any()`

```python
print(np.all([True, True, False])) # and operation
print(np.any([True, True, False])) # or operation
```

`mean(), median(), std()`

```python
x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])

x.mean()

np.median(x)
np.median(y, axis=-1) # [2, 5]

x.std()

```

#### Element-wise operation and broadcast
- Vectorizing: 루프를 사용하지 않고 배열의 각 요소를 계산. 코드 가독성이 높아지고 성능도 좋아짐.
- Broadcast: 서로 다른 Shape을 가진 배열의 산술 연산을 위해 NumPy가 가진 규칙

`Broadcasting Rule`
1. 차원 통일: 우측 (높은 축 번호) Align, 확장된 축은 요소 수 1로 생성
    - 차원 통일 원리: (3, ) 은 (1, 3), (1, 1, 3), (1, 1, 1, 3) 으로도 볼 수 있다.
2. 축 간 Item 수 통일: Item 수가 1인 배열이 상대편 동일 축 Item 수만큼 확장

```
Q1)
(3, 1) + (3, ) -> (3, 3) + (3, 3)

Q2)
(2, 3, 5) + (3, 5) -> (2, 3, 5) + (2, 3, 5)

Q3)
(1, 3, 5) + (2, 1, 5) -> (2, 3, 5) + (2, 3, 5)

Q4)
(3, 2) + (3,) -> ValueError

Q5)
(4, 4) + (2, ) -> ValueError
```


#### Array Manipulation
1. Flattening (다차원 배열을 1차원 배열로 전환) `np.ndarray.ravel()`
2. Reshaping (배열을 원하는 Shape로 변경) `np.ndarray.reshape()`