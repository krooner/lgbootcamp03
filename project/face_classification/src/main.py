from ai_chat import *
from ask_music_preference import *
from face_module import recognize_emotion
from music_player import *
from smtp import send_email_emotion, send_email_emotion_statistics
import glob

if __name__ == "__main__":
    before_emotion, before_emotion_prob = recognize_emotion(detection_duration=10., before=True)

    if before_emotion == None:
        print("Failed to detect emotion")
        exit(0)

    # Function 1: Sending Email for light automation
    send_email_emotion(before_emotion)

    # Function 2: asking answer for music recommendation
    is_recommended = ask_music_preference(before_emotion)
    if is_recommended: # If True, play music
        music_player(before_emotion)
    else: # If False, start conversation with AI Speaker
        ai_chat(before_emotion, turn_count=3)

    after_emotion, after_emotion_prob = recognize_emotion(detection_duration=10., before=False)

    # send an email containing emotion statistics
    print(f"""
        Emotion before: {before_emotion}
        Emotion After: {after_emotion}
    """)
    send_email_emotion_statistics(before_emotion, before_emotion_prob, after_emotion, after_emotion_prob)

    # Remove all saved images
    for file_loc in glob.glob(os.path.join(os.getcwd(), "../images/*.png")):
        os.remove(file_loc)