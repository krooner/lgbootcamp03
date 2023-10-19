# Day 4

## 3. DL Basic

### Multi-layer Perceptron

**Input gate logic**

- Perceptron은 단순한 선형 분류기
- Multi Layer Perceptron을 사용해야 함
    1. 단순한 기능 구현에도 많은 Parameter update가 필요 -> Perceptron에 대한 인기 급격히 감소 (AI Winter)
    2. Backpropagation
        - Output layer로부터 역방향으로 기울기 계산을 진행하는 방법
        - Feedforward로 Loss 구한 후 Backward 방향으로 전파시키면서 학습: 연산량 대폭 감소


### MNIST Dataset and Keras