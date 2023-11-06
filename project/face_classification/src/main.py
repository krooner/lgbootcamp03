from face_module import recognize_emotion 
from ai_chat import ai_chat
from ask_music_preference import chat_start
from music_player import music_player
from smtp import send_email

if __name__ == "__main__":
    before_result = recognize_emotion(detection_duration=10.)

    # Function 1: Sending Email for light automation
    send_email(before_result)

    # Function 2: asking answer for music recommendation
    is_recommended = chat_start(before_result)
    if is_recommended: # If True, play music
        music_player(before_result)
    else: # If False, start conversation with AI Speaker
        ai_chat(before_result, turn_count=3)

    after_result = recognize_emotion(detection_duration=10.)

    print(f"""
        Emotion before: {before_result}
        Emotion After: {after_result}
    """)

