# Day 3

## Advanced CNN

### AlexNet and VGGNet

#### LeNet
- `Gradient-Based Learning Applied to Document Recognition (1998)`
- 모델 검증을 위해 제작한 데이터셋이 MNIST Handwrite DB

#### AlexNet
- ILSVRC 12' 에서 우승한 CNN 모델로 16.4%의 Error rate을 기록하며 2nd place의 25.5%에 크게 앞섬.
- 처음으로 ReLU를 적용하였음
- LRN (Local Response Normalization): 1st-conv와 2nd-conv 사이에 ReLU를 적용 후 Max-pooling 전에 LRN 사용
- Data Augmentation, Dropout, Weight decay 적용
- Dynamic Learning rate (초기에는 1e-2로 적용하다가 1e-10까지 점차 감소)
- GTX580 두 장으로 1주일 이상 병렬 학습 

#### VGGNet
- ILSVRC 14'에서 2nd place를 기록했지만 간단한 구조와 훌륭한 성능을 보임
- `3x3` Convolutional filter를 적용
    - `5x5, 7x7` 대신, `3x3`을 여러번 써도 같은 효과가 남
    - 학습 Parameter 수가 줄어들고 이미지 크기 감소 또한 적어 Layer 증가에 유리

**학습 방식**
- Data augmentation으로 입력 데이터 생성
- Training scale `S`에 따라,
    1. 원본 데이터를 `S`에 맞춰 Reshape: `1024 x 512` $\rightarrow$ `512 x 256` (`S``=2)
    2. `224x224` 크기로 Random crop
- `S` 정하는 방식
    1. Single-scale Training: `S=256 or 384`. 먼저 256 기준으로 학습 후 384로 재학습 
    2. Multi-scale Training: `256<=S<=512`. 먼저 Single-scale Training 방식으로 학습 후 해당 방식으로 재학습

**평가 방식**
- Data augmentation으로 입력 데이터 생성하며 Training시 `S`, Testing시 `Q` 사용 (의미는 동일)

### GoogLeNet

22개의 Layer로 구성된 DNN으로, 효율을 극대화한 Inception Module을 제시해서 ILSVRC 14' 우승
- 기존 DNN의 문제점: Model이 Deep해지고 Layer가 넓을수록 성능이 향상되지만, 연산량이 증가하고 과적합 가능성이 높아짐
- 제안 방식: 전체적으로 Network 내 연결을 줄이고 (Sparse) 세부 행렬 연산은 Dense하게 처리하면 효과적일 것이다.

#### Inception Module

Convolution layer를 Sparse하게 연결하고 행렬 연산은 Dense하게 처리하는 구조로, 여러 개의 Layer를 한 층으로 구성하는 형태
- Feature를 효과적으로 추출하기 위해 `1x1, 3x3, 5x5`의 Convolution과 `3x3`의 Max-pooling 사용
- Output을 Channel-wise Concatenation하여 Feature map을 구성

**초기 Inception Module**
- 문제점: 계산량이 너무 많음
- 개선책: `1x1` Convolution을 적용 (Bottleneck)
    - Dimension reduction 기능을 수행하여, 학습 Parameter 수가 감소된다.
    - Feature map 내에는 중복 또는 미사용되는 것들이 존재하는데, 해당 Bottleneck을 통과하면서 Channel이 감소했다가 다시 증가함.
    - Channel을 크게 줄이지 않는 한 Bottleneck은 성능에 큰 영향을 미치지 않음. 

**GoogLeNet** (`Inception.v1`)
- 모델 내 9개의 Inception module가 존재함.
- 마지막은 Classifier: Fully-connected layer를 대부분 제거했으며, Parameter가 줄어도 잘 동작함
- 중간에 Auxiliary classifier가 존재: 추가적인 Gradient를 얻어 중간 Layer 학습을 도우며, 학습 후 제거됨

**개선 작업** (`Inception.v{1, 2}`)
- `5x5` Convolution을 `3x3 + 3x3` 으로 변경하여 연산량 28% 감소
- `n x n`을 `1 x n + n x 1` 로 변경하여 연산량 33% 감소
- Auxiliary classifier_1 을 제거: 성능에 큰 영향을 주지 못함
- Optimizer를 RMSprop으로 변경
- Label smoothing 사용: 0, 1이 아닌 0.1, 0.9와 같이 변형
- Factorized `7x7`: 맨 앞단 `7x7` Convolution을 `3x3` Convolution 2개로 변경
- BN-auxiliary: 마지막 Fully-connected layer에 Batch normalization 적용
- `Inception.v4`는 직전 버전을 변형 및 실험

### ResNet
VGG-19 구조에 Convolution layer와 Shortcut을 추가함

#### Residual block
- Layer가 많을 수록 무조건 성능이 좋아지진 않는다: Vanishing/Exploding gradient
- 기존 방식 (input `x` and output `H(X)`)
    - `H(x)`가 학습 대상
- Residual block (input `x` and output `H(x) = F(x) + x`)
    - 입력값에 출력값을 더해줄 수 있는 Shortcut을 추가
    - `F(x) = H(x) - x = Residual (잔차)`가 학습 대상
    - Layer가 많아져도 최적화가 쉬워지고 정확도가 향상됨

### WideResNet, DenseNet, PyramidNet

|Model|Character||
|---|---|---|
|ResNet|Layer를 깊게|Residual block|
|WideResNet|Layer를 넓게|깊이 증가보다 Residual block의 개선|
|DenseNet|Layer 간 Skip connection을 많이|Feedforward 방식으로 각 Layer를 다른 모든 Layer에 연결|
|PyramidNet|Layer 간 Feature map 변화|모든 Layer에서 #Channel을 변경하여 Channel이 급변하면서 발생할 수 있는 성능 저하를 방지|

### CAM (Class Activation Map)
- CNN 모델은 Labeled data를 활용한 Supervised Learning (Labeling... expensive!)
    - 일반적으로 Flatten에서 Fully-connected layer로 넘기는 순간 Filter가 갖는 위치 정보가 소실됨..
- Model이 이미지의 어떤 부분을 보고 분류하는지 알 수 있게 시각화
- 위치 정보 보존을 위해 Flatten 대신, GAP (Global Average Pooling) 사용
    - 성능 저하가 크지 않고, 추가 Convolution layer를 추가하여 성능 저하 해소 가능
- Label만으로 학습시켰지만 Classification은 물론 객체 위치도 파악 가능
    - Weakly supervised learning

#### Limitation
1. Flatten을 GAP으로 대체해야 함: 처음부터 GAP 구조였다면 대체 불필요
2. GAP 직전의 Convolution layer를 통해서만 CAM을 얻을 수 있음: 다른 Layer는 불가
3. GAP 뒷 단 Dense layer의 Weight update를 위해, Fine-tuning이나 Re-training 과정이 필요: 처음부터 GAP 구조였다면 학습 불필요

#### Gradient-weighted CAM
- GAP에 의존하지 않고 Gradient를 이용함
- Flatten을 GAP으로 대체할 필요 없음
- GAP 직전 Convolution layer뿐만 아니라 다른 위치의 Convolution layer에도 적용할 수 있음