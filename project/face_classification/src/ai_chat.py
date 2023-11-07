import openai
import random
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import sounddevice # 코드 내에는 안써도 이게 import 되어야 raspberry pi에서 에러가 안남
from speaker import text_to_speech_and_play, play_sound_effect, start_listening_file_loc, finish_listening_file_loc
from openai_api_key import api_key

# API 키 설정
openai.api_key = api_key

# 음성 인식 객체
r = sr.Recognizer()

# 대화가 길어지면 그동안 했던 대화들을 요약해서 openai 서버에 전달
# def summarize(answer):
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{
#             'role': 'user',
#             'content': f'이 내용 한국어로 한 문장으로 요약해줘 ###\n{answer}\n###'
#         }],
#     )

#     return completion.choices[0].message.content

# GPT-3.5 api를 이용하여 챗봇과 대화
def ask_gpt(question):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=question
        )
        return completion.choices[0].message
    except:
        return

def ai_chat(emotion, turn_count):
    # 사용자로부터 입력 받아 GPT-4에 질문하고 응답 출력
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    emotions_kr = ["화나", "짜증나", "무서워", "행복해", "슬퍼", "놀라", "그저그래"]
    idx = emotions.index(emotion)

    turns = turn_count

    # ChatGPT에게 역활 부여. 아래 문자열을 수정하면 말투가 바뀔 수 있음
    gpt_role = "You are a kind and friendly counselor. You don't have to talk formally but use 존댓말. If 'last_turn' == True, answer 'message' and finish the chat."
    start_with = f"I am currently {emotion}. Ask me what happened. Start like '오늘 {emotions_kr[idx]}보여요' with appropriate expression in Korean"
    answer_briefly = "'condition': Please provide a reply within 30 tokens."
    first_input = [{"role": "system", "content": gpt_role + start_with}]
    response = ask_gpt(first_input)
    print("알스:", response["content"])
    text_to_speech_and_play(response["content"])

    messages = [{"role": "system", "content": gpt_role}]
  
    # while 문이 종료될때 까지 대화
    # while 문 종료 조건은 대화 상대가 "대화 종료"라고 말하기
    while (turns):
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
        #audio = input("입력: ")
        if audio:
            try:
                # summarized = summarize(user_input)
                if len(messages) >= 6:
                    messages = [messages[0]] + messages[len(messages)-4:]
                new_input = r.recognize_google(audio, language='ko-KR')
                #new_input = audio
                print("사용자: ", new_input)
                # 대화 종료 로직
                if ("대화종료" in new_input or "대화 종료" in new_input):
                    break
                turn_message = "'last_turn': {}, ".format(False if turns > 1 else True)
                msg = turn_message + "'message': " + new_input + ", " + answer_briefly
                messages.append({"role": "user", "content": msg})
                response = ask_gpt(messages)
                if response:
                    messages.append({"role": "system", "content": response["content"]})
                    print("알스:", response["content"])
                    text_to_speech_and_play(response["content"])
                    turns -= 1
                else:
                    print("오류가 발생했습니다. 다시 말씀 해주세요.")
                    text_to_speech_and_play("오류가 발생했습니다. 다시 말씀 해주세요.")
            # 말을 이해 못했을 때 에러
            except sr.UnknownValueError:
                print("알아듣지 못했어요")
                text_to_speech_and_play("알아듣지 못했어요")
            # 서버 에러
            except sr.RequestError:
                print("서버 에러")
                text_to_speech_and_play("서버에 문제가 있어요")
