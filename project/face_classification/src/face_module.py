from statistics import mode
import time
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import Counter
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = "../trained_models/aihub_aug.tflite"

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}
enko_dict = {
    'Angry': "화남", 'Disgust': "짜증남", 'Fear': "무서움", 'Happy': "행복함", 'Sad': "슬픔", 'Surprise': "놀람", 'Neutral': "무표정"
}

# loading face detection model
face_detection = load_detection_model(detection_model_path)

# loading emotion recognition model
interpreter = tflite.Interpreter(model_path=emotion_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# getting input model shapes for inference
IMG_SHAPE = tuple(input_details[0]['shape'][1:3])

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)


def recognize_emotion(detection_duration=10., before=True):

    # starting video streaming
    cv2.namedWindow('window_frame')
    camera = cv2.VideoCapture(0)

    # set frame width and height for resource saving
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    first_detection_time = None
    detection_results = []
    # emotion_window = []

    prob_sum = np.zeros(7)
    each_prob_dict = dict()

    most_freq_emotion = None

    while True:
    
        if first_detection_time != None:
            if time.time() - first_detection_time > detection_duration:
                print("감정 인식 완료.")
                try:
                    most_freq_emotion = Counter(detection_results).most_common(1)[0][0]

                    # print(Counter(detection_results))
                    print(f"사용자의 주된 감정은 '{enko_dict[Counter(detection_results).most_common(1)[0][0]]}' 입니다.")
                except IndexError as e:
                    print("No result exists.")
                    return None
                break

        ret, bgr_image = camera.read()
        if ret != True:
            print("Video capture Error.")
            break

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        emotion_text = None

        for face_coordinates in faces:
            if first_detection_time == None: 
                first_detection_time = time.time()
                print(f"최초 얼굴 감지. {int(detection_duration)}초간 감정 인식 실행 중...")

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, dsize=IMG_SHAPE)
            except:
                continue

            gray_face = preprocess_input(gray_face, False)
            gray_face = np.expand_dims(gray_face, axis=0)
            gray_face = np.expand_dims(gray_face, axis=-1)

            interpreter.set_tensor(input_details[0]['index'], gray_face)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            prob_sum += output_data.squeeze(0)

            emotion_probability = np.max(output_data)
            emotion_label_arg = np.argmax(output_data)
            emotion_text = emotion_labels[emotion_label_arg]
            # print(emotion_text)
            
            detection_results.append(emotion_text)

            # emotion_window.append(emotion_text)
            # if len(emotion_window) > frame_window:
            #     emotion_window.pop(0)
            # try:
            #     emotion_mode = mode(emotion_window)
            # except:
            #     continue

            if emotion_text == 'Angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'Sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'Happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'Surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_text,
                    color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)

        if emotion_text != None:
            each_prob_dict[emotion_text] = bgr_image

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()


    img_folder_dir = os.path.join(os.getcwd(), "../images")
    if not os.path.exists(img_folder_dir):
        os.mkdir(img_folder_dir)

    cv2.imwrite(os.path.join(img_folder_dir, f"{'before' if before else 'after'}_image.png"), each_prob_dict[most_freq_emotion]) 

    # emotion, probabilities, image_of_emotion (save in advance)
    return most_freq_emotion, prob_sum / len(detection_results) 
