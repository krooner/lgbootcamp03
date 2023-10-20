# Day 4

## 3. DL Basic

### Multi-layer Perceptron
- Perceptron은 단순한 선형 분류기
- Multi Layer Perceptron을 사용해야 함
- 단순한 기능 구현에도 많은 Parameter update가 필요 -> Perceptron에 대한 인기 급격히 감소 (AI Winter)

#### Backpropagation
- Output layer로부터 역방향으로 기울기 계산을 진행하는 방법
- Feedforward로 Loss 구한 후 Backward 방향으로 전파시키면서 학습: 연산량 대폭 감소

### MNIST Dataset and Keras
MNIST (Modified National Institute of Standards and Technology database)

#### Train data, test data, and validation data
1. Train data: 모델의 Optimal parameter를 찾기 위한 학습 과정에서 사용하는 데이터
2. Test data: 학습이 잘 되었는지 확인하기 위한 평가 과정에서 사용하는 데이터
3. Validation data: 학습에는 사용하지 않지만, 과적합 여부 확인을 위한 데이터
|**Topic**|Introduction to ML/DL|Linear Regression|Classification|DL Basic|CNN||**Topic**|Introduction to ML/DL|Linear Regression|Classification|DL Basic|CNN|

#### Underfitting and overfitting
1. Underfitting: 학습이 부족한 상황. 추가 학습 시 Train & test data에 대한 오차 모두 개선 가능
2. Overfitting (과적합): 특정 데이터에 과도하게 학습이 된 상황. 모델 Parameter가 학습 데이터에 과도하게 맞춰진 상황. 학습 데이터에 대한 정확도는 높으나, 평가 데이터에 대한 정확도는 낮아지는 경향.

#### Mini-batch
1. Batch Gradient Descent
    - 전체 데이터가 곧 Batch
    - 학습 시 전체 데이터에 대한 Cost 평균을 기준으로 기울기 계산
2. Mini-batch Gradient Descent
    - 전체 데이터를 일정 갯수의 묶음들로 나누고, 각 묶음이 Batch
    - Batch 데이터를 기준으로 Cost 및 기울기 계산
3. Stochastic Gradient Descent
    - 각 데이터 단위로 처리 및 Cost와 기울기 계산

#### tensorflow.keras
keras에서 제공하는 다양한 API를 이용하여 모델 개발
- Sequential API
- Functional API
- Custom Model (Subclassing API)

```python
# Preparing Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

# Defining Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compiling Model
model.compile(
    optimizer="SGD",
    loss="sparse_categorica;_crossentropy",
    metrics=["accuracy"]
)

# Training
model.fit(
    x_train, y_train, epochs=5
)

# Evaluating
model.evaluate(
    x_test, y_test, verbose=2
)

```