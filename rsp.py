import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import random
from keras.layers import TFSMLayer
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
try:
    model = TFSMLayer("model.savedmodel", call_endpoint='serving_default')
except Exception as e:
    print("Error loading model:", e)
    exit()

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # Spacebar is pressed (ASCII for spacebar is 32)
    if keyboard_input == 32:
        # Make the image a numpy array and reshape it to the model's input shape
        image_normalized = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_normalized = (image_normalized / 127.5) - 1

        try:
            # Predicts the model
            prediction = model(image_normalized)

            # Assuming prediction is a dictionary
            if isinstance(prediction, dict):
                prediction = list(prediction.values())[0]

            prediction = prediction.numpy()  # Ensure prediction is a numpy array
            print("Prediction array:", prediction)  # Print the prediction array to debug
            index = np.argmax(prediction[0])
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            print("user:", class_name, end="")
            print(" accuracy: ", str(np.round(confidence_score * 100, 2)), "%")

            array = ['paper', 'rock', 'scissors']
            computer = random.choice(array)
            player = class_name

            print(f'computer = {computer}, player = {player}')

            if computer == player:
                print('tie')
            elif (computer == 'scissors' and player == 'rock') or (computer == 'rock' and player == 'paper') or (
                    computer == 'paper' and player == 'scissors'):
                print('player win')
            else:
                print('computer win')

        except Exception as e:
            print("Error during prediction:", e)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
