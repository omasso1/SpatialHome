import mediapipe as mp
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarkerResult
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from enum import Enum
from matplotlib import pyplot as plt
import math
import time

fileDir = Path(__file__).resolve().parent

class HandLandMarks(Enum):
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

class PoseLandmark(Enum):
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32


BaseOptions = mp.tasks.BaseOptions

HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class LandMarkRecognizer():
    def __init__(self, live_call_back: function) -> None:
        model_path = str(Path(fileDir, 'models', 'holistic_landmarker.task'))
        options = HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                min_face_detection_confidence=0.5,
                min_pose_detection_confidence=0.5,
                min_hand_landmarks_confidence=0.5,
                result_callback=self.print_result)

        self._detector:HolisticLandmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(options)

    def print_result(self,
                     result: HolisticLandmarkerResult,
                     output_image: mp.Image,
                     timestamp_ms: int):
        print('holistic landmarker result: {}'.format(result))

    def detect_async(self, image: mp.Image, timestamp: int):
        self._detector.detect_async(image, timestamp)


