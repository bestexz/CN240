from os import listdir
import os
import mediapipe as mp
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

path = r"train_images/"
namelist = listdir(path)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for images in namelist:
        image = cv2.imread(path + images, 0)

        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )

        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
        #                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),
        #                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                              )

        print(images)
        cv2.imshow('image', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cv2.destroyAllWindows()
print(results.face_landmarks.landmark[0].visibility)

num_coords = len(results.face_landmarks.landmark)
print(num_coords)

landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val), ]

print(landmarks)

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

    