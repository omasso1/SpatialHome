import mediapipe as mp
import os
from pathlib import Path
from matplotlib import pyplot as plt
import math
from HandRecognizer import HandRecognizer
import time
from mainApp import MainApp

fileDir = Path(__file__).resolve().parent

# recognizer = HandRecognizer()
# image_path = str(Path(fileDir, 'images' , 'not_whole_body.jpg'))
# # image_path = str(Path(fileDir, 'images' , 'manThumbsUp.png'))

# result = recognizer.detect_hands(image_path)
# recognizer.show_image(image_path, result)

app = MainApp()
app.run()



