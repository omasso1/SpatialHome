import mediapipe as mp
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarkerResult
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from enum import Enum
from matplotlib import pyplot as plt
import math
import time

fileDir = Path(__file__).resolve().parent

class LandMarks(Enum):
    WRIST = 0,
    THUMB_0 = 1,
    THUMB_1 = 2,
    THUMB_2 = 3,
    THUMB_3 = 4,
    INDEX_0 = 5,
    INDEX_1 = 6,
    INDEX_2 = 7,
    INDEX_3 = 8,
    MIDDLE_0 = 9,
    MIDDLE_1 = 10,
    MIDDLE_2 = 11,
    MIDDLE_3 = 12,
    RING_0 = 13,
    RING_1 = 14,
    RING_2 = 15,
    RING_3 = 16,
    PINKY_0 = 17,
    PINKY_1 = 18,
    PINKY_2 = 19,
    PINKY_3 = 20

class HandRecognizer:
    def __init__(self):
        model_path = str(Path(fileDir, 'models', 'holistic_landmarker.task'))
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HolisticLandmarkerOptions(base_options=base_options)

        self._detector = mp.tasks.vision.HolisticLandmarker.create_from_options(options)


    def detect_all(self, image_path: str) -> HolisticLandmarkerResult:
        image = mp.Image.create_from_file(image_path)
        t = time.time()
        result = self._detector.detect(image)
        print(time.time() - t)
        return result

    def _display_one_image(self, image, title, plot, titlesize=16):
        plt.plot(plot)
        plt.imshow(image)
        if len(title) > 0:
            plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))


    def show_image(self, image_path: str, landmarks: HolisticLandmarkerResult):
        plt.rcParams.update({
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'xtick.labelbottom': False,
            'xtick.bottom': False,
            'ytick.labelleft': False,
            'ytick.left': False,
            'xtick.labeltop': False,
            'xtick.top': False,
            'ytick.labelright': False,
            'ytick.right': False
        })

        mp_hands = mp.tasks.vision.HandLandmarksConnections
        mp_pose = mp.tasks.vision.PoseLandmarksConnections
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_drawing_styles = mp.tasks.vision.drawing_styles

        image = mp.Image.create_from_file(image_path)
        image_copy = image.numpy_view().copy()

        mp_drawing.draw_landmarks(
            image_copy,
            landmarks.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        mp_drawing.draw_landmarks(
            image_copy,
            landmarks.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        mp_drawing.draw_landmarks(
            image_copy,
            landmarks.pose_landmarks,
            mp_pose.POSE_LANDMARKS,
            mp_drawing_styles.get_default_pose_landmarks_style(),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        plt.imshow(image_copy)
        plt.show()
        tmp = 10
