from statistics import mode
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

model_path = "../trained_models/aihub_trans_aug.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height, width = input_details[0]['shape'][1:3]

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# emotion_model_path = '../trained_models/aihub_trans_aug.h5'
emotion_labels = {
    0: 'Angry', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy',
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
# emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = (height, width)

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    # time.sleep(1)
    ret, bgr_image = video_capture.read()
    if ret != True:
        print("Video capture Error.")
        break

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, axis=[0, -1])

        interpreter.set_tensor(input_details[0]['index'], gray_face)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        # emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(output_data)
        emotion_label_arg = np.argmax(output_data)
        emotion_text = emotion_labels[emotion_label_arg]
        print(emotion_text)

        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

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
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
