# README

## 준비물
1. **Raspberry Pi (Raspi)**
2. Raspi에 연결된 **웹캠**
3. Raspi에 연결된 **마이크**
4. Raspi에 연결된 **스피커**

5. iOS Automation 기능을 지원하는 **아이폰**
6. 스마트폰과 같은 Wi-Fi에 있는 **IoT 기기**

## 실행

```bash
$ git clone https://github.com/krooner/lgbootcamp03.git
$ cd lgbootcamp03/project
# 필요 라이브러리 설치: 오디오 데이터 활용에 필요한 STT 및 TTS 기능 및 OpenAI 라이브러리를 설치
$ pip install -r requirements.txt


$ cd ./src
# ssh로 실행할 경우, Raspiberry Pi에서 감정 인식 화면 출력
$ export DISPLAY=:0.0 

$ python3 main.py
```

## 사전 준비 데이터

### 1. 음악 파일 준비하기
감정 기반 음악 재생 기능을 수행하기 위해서는 lgbootcamp03/project/music 폴더를 아래와 같이 구성
1. `./music` 폴더에 7개 하위 폴더를 만든다.
    - `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised`
2. 각 하위 폴더에 재생할 음악 `.mp3` 파일들을 저장한다.
    - 파일명에 한글이 들어가면 오류 발생할 수 있음 `UnicodeEncodeError`

음성 인식 시작 및 종료 음성은 `./sound` 폴더에 위치한다.
1. 음성 인식 시작음 `./sound/siri-sound-effect_finish-listening.mp3`
2. 음성 인식 종료음 `./sound/siri-sound-effect_start-listening.mp3`

### 2. OpenAI API 및 SMTP Key 설정하기
대화형 AI 및 이메일 전송에 필요한 OpenAI API와 smtplib을 활용하기 위해서는 아래와 같이 설정

1. `./src/open_api_key.py`: OpenAI API를 사용하기 위한 Key
    - `api_key`: OpenAI API Key
2. `./src/email_key.py`: IoT 기기 제어 시, 이메일 트리거 방식을 위해 필요한 발신자 이메일 계정 정보
    - `msg_from`: 발신자 이메일
    - `msg_to`: 수신자 이메일
    - `email_id`: 발신자 계정 ID
    - `email_pw`: 발신자 계정 PW
