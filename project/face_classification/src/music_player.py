import os
import random
import pygame
import time

music_path = "../music/"
pygame.mixer.init()

def music_player(emotion):
    # 음악이 저장된 폴더로 경로를 설정합니다.
    folder_path = os.path.join(music_path, emotion)
    
    # 해당 폴더에 있는 모든 mp3 파일의 리스트를 가져옵니다.
    try:
        songs = [song for song in os.listdir(folder_path) if song.endswith('.mp3')]
    except FileNotFoundError:
        print(f"The directory for {emotion} does not exist.")
        return
    
    # 랜덤하게 하나의 곡을 선택합니다.
    if songs:
        song_to_play = random.choice(songs)
        song_path = os.path.join(folder_path, song_to_play)
        
        # 선택된 노래를 재생합니다.
        print(f"Playing {song_to_play} from {emotion} category.")
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        
        # 재생 중지를 위한 입력 대기
        input("Press 'Enter' to stop playing...")
        time.sleep(10)
        pygame.mixer.music.stop()
    else:
        print("No MP3 files found to play.")

music_player("Angry")