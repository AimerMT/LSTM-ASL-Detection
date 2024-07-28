import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("lstm-hand-model.h5")
lm_list = []
label = ""
neutral_label = ""

def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    for hand_landmarks in results.multi_hand_landmarks:
        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    return frame

def draw_bounding_box_and_label(frame, results, label):
    for hand_landmarks in results.multi_hand_landmarks:
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)
        h, w, c = frame.shape
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)
        color = (0, 0, 255) if label != neutral_label else (0, 255, 0)
        thickness = 2 if label != neutral_label else 1
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(frame, f"Status: {label}", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    return frame

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    percentage_result = result * 100
    print(f"Model prediction result: {percentage_result}")
    if result[0][0] > 0.9:
        label = "A"
    elif result[0][1] > 0.9:
        label = "B"
    elif result[0][2] > 0.9:
        label = "C"
    elif result[0][3] > 0.9:
        label = "D"
    elif result[0][4] > 0.9:
        label = "E"
    elif result[0][5] > 0.9:
        label = "F"
    elif result[0][6] > 0.9:
        label = "G"
    elif result[0][7] > 0.9:
        label = "H"
    elif result[0][8] > 0.9:
        label = "I"
    elif result[0][9] > 0.9:
        label = "J"
    elif result[0][10] > 0.9:
        label = "K"
    elif result[0][11] > 0.9:
        label = "L"
    elif result[0][12] > 0.9:
        label = "M"
    elif result[0][13] > 0.9:
        label = "N"
    elif result[0][14] > 0.9:
        label = "O"
    elif result[0][15] > 0.9:
        label = "P"
    elif result[0][16] > 0.9:
        label = "Q"
    elif result[0][17] > 0.9:
        label = "R"
    elif result[0][18] > 0.9:
        label = "S"
    elif result[0][19] > 0.9:
        label = "T"
    elif result[0][20] > 0.9:
        label = "U"
    elif result[0][21] > 0.9:
        label = "V"
    elif result[0][22] > 0.9:
        label = "W"
    elif result[0][23] > 0.9:
        label = "X"
    elif result[0][24] > 0.9:
        label = "Y"
    elif result[0][25] > 0.9:
        label = "Z"

    return str(label)


cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#   cv2.resizeWindow("image", 1000, 600)  

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    crop_size = 0.8
    x_center = w // 2
    y_center = h // 2
    crop_w = int(w * crop_size)
    crop_h = int(h * crop_size)

    x1 = x_center - crop_w // 2
    x2 = x_center + crop_w // 2
    y1 = y_center - crop_h // 2
    y2 = y_center + crop_h // 2
    cropped_frame = frame[y1:y2, x1:x2]

    frame_resized = cv2.resize(cropped_frame, (w, h))
    frameRGB = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []
            frame = draw_landmark_on_image(mpDraw, results, frame_resized)
            frame = draw_bounding_box_and_label(frame_resized, results, label)
        cv2.imshow("image", frame_resized)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
