# README

## 준비사항

### 필요 라이브러리 설치
오디오 데이터 활용에 필요한 STT 및 TTS 기능 및 OpenAI 라이브러리를 설치해야 한다.
  
```bash
$ pip install -r requirements.txt
```

### 음악 파일 준비하기
1. `./music` 폴더에 7개 하위 폴더를 만든다.
    - `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised`
2. 각 하위 폴더에 재생할 음악 `.mp3` 파일들을 저장한다.
    - 파일명에 한글이 들어가면 오류 발생할 수 있음 `UnicodeEncodeError`

### API Key 설정하기
대화형 AI 및 이메일 전송에 필요한 OpenAI API와 smtplib을 활용하기 위해서는 아래와 같이 설정해야 한다.

1. `./src/open_api_key.py`: OpenAI API를 사용하기 위한 Key
    - `api_key`: OpenAI API Key
2. `./src/email_key.py`: IoT 기기 제어 시, 이메일 트리거 방식을 위해 필요한 발신자 이메일 계정 정보
    - `msg_from`: 발신자 이메일
    - `msg_to`: 수신자 이메일
    - `email_id`: 발신자 계정 ID
    - `email_pw`: 발신자 계정 PW