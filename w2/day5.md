# Day 5

## TensorFlow Lite
1. 모델 선택: 새 모델을 만들어 학습시키거나, 이미 학습된 모델을 선택
2. 변환: TFLite Converter를 이용하여 `.tflite` 파일로 변환
3. 실행: 모바일 및 임베디드 기기에서 `.tflite` 파일을 로드 후 활용
4. 최적화: 양자화 (Quantization) 를 통해 모델 크기를 축소하고 속도를 높임
    - 양자화란 float32를 float16 또는 int8로 변환하는 것으로, 매우 작은 정확도는 포기
