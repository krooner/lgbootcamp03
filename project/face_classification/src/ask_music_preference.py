import speech_recognition as sr
from gtts import gTTS
import pygame
from speaker import text_to_speech_and_play, play_sound_effect, start_listening_file_loc, finish_listening_file_loc
import os

r = sr.Recognizer()

def ask_music_preference(emotion):
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    texts = [
        "화난 기분을 풀어줄 ",
        "짜증나는 기분을 풀어줄 ",
        "무서울 때 듣기 좋은",
        "행복할 때 듣기 좋은 ",
        "슬픈 마음을 위로해줄 ",
        "놀란 마음을 가라앉힐 ",
        "평소에 듣기 좋은"
    ]
    idx = emotions.index(emotion)
    text = texts[idx] + "노래를 틀어드릴까요?"
    print("알스: " + text)
    text_to_speech_and_play(text)
    
    while True:
        with sr.Microphone() as source:
            play_sound_effect(start_listening_file_loc)
            print("음성 인식 시작")
            try:
                audio = r.listen(source, timeout=3, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                print("Timeout: 음성 인식이 감지되지 않았습니다.")            
                audio = None
            play_sound_effect(finish_listening_file_loc)
            print("음성 인식 종료")
        # audio = input("입력: ")

        if audio:
            try:
                new_input = r.recognize_google(audio, language='ko-KR')
                # new_input = audio
                print("사용자: ", new_input)
                # 대화 종료 로직
                if ("응" in new_input or "그래" in new_input or "네" in new_input or "틀어" in new_input):
                    return True
                elif ("아니" in new_input or "괜찮아" in new_input or "아니오" in new_input or "틀지마" in new_input):
                    return False
                else:
                    print("다시 말씀해주세요")
                    text_to_speech_and_play("다시 말씀해주세요")
            # 말을 이해 못했을 때 에러
            except sr.UnknownValueError:
                print("알아듣지 못했어요")
                text_to_speech_and_play("알아듣지 못했어요")
            # 서버 에러
            except sr.RequestError:
                print("서버 에러")
                text_to_speech_and_play("서버에 문제가 있어요")