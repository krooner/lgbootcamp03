import os
from gtts import gTTS
import pygame

def text_to_speech_and_play(text):
    tts = gTTS(text=text, lang="ko", slow=False)

    output_filename = "temp_output.mp3"
    tts.save(output_filename)

    pygame.mixer.init()
    pygame.mixer.music.load(output_filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    
    os.remove(output_filename)

def play_sound_effect(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    return 