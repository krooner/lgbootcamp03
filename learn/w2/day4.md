# Day 4

## Generative Model

정해진 데이터를 학습한 결과로 새로운 데이터를 생성하는 모델로, 대량의 데이터를 사용하여 진짜같은 이미지, 텍스트, 음성 데이터를 생성한다.

### Autoencoder

입력 데이터 `x`가 Label 역할을 수행하는 Unsupervised learning

#### Latent space and dimension reduction

일반적으로 `x`를 더 작은 차원의 `z`로 Encoding하며, 주요 Feature는 보존되나 일부 정보는 유실 가능. `x`를 제대로 Encoding하고 `z`를 제대로 Decoding하려면?
1. Encoder: 입력값을 Latent space에 Mapping
2. Latent space:
3. Decoder: Latent space vector를 출력으로 Mapping

### VAE (Variational Autoencoder)

값이 아닌, 분포 (평균과 표준편차) 로 Encoding하고 생성 시에는 분포에서 Sampling 후 Decoding
- Autoencoder는 데이터를 잘 표현하는 직접적인 값으로 Encoding
- VAE는 데이터를 잘 표현하는 $\mu, \sigma$의 확률분포로 Encoding
- 얼굴 이미지의 경우 $\mu, \sigma$의 변화에 따라 얼굴 방향, 표정, 안경 등 특징들이 미세하게 달라짐 

### GAN

#### Generator and Discriminator
- Generator (생성자): 학습이 진행되면서, 진짜 같은 이미지를 더 잘 생성
    - Discriminator의 판정 결과를 기준으로 학습
- Discriminator (감별자): 학습이 진행되면서, 진짜와 가짜를 더 잘 구별하게 됨
    - Real sample과 Fake sample 학습 반복
- Discriminator가 더 이상 Real과 Fake를 구분하지 못하는 상황이 될 때까지 학습 진행