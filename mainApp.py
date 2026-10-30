from ursina import Ursina, load_blender_scene, EditorCamera, Mesh, Entity, Vec3, color
from ursina.shaders import lit_with_shadows_shader
from pathlib import Path
from typing import cast
import time
import cv2
from numpy import ndarray
from LandMarkRecognizer import LandMarkRecognizer
from mediapipe import Image
import mediapipe as mp
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarkerResult


fileDir = Path(__file__).resolve().parent

class MainApp():
    def __init__(self):
        self.camera_address = "http://192.168.1.134:8080/stream"
        self._video_cap = self._init_camera()
        self._ursina_app = self._init_ursina()
        self._land_mark_recognizer = LandMarkRecognizer(self._process_mp_result)
        self.frame_rate = 120
        self.frame_time = 1 / self.frame_rate
        self._show_camera = True
        self._result_processed = True
        self._result_image = None

    def _init_ursina(self) -> Ursina:
        ursina_app =  Ursina()
        self._room = load_blender_scene("room", fileDir / "roomModel", reload=True, vertex_colors=True)
        self._human_model = Entity(
            model=Mesh(
                vertices=[],
                mode='line',
                thickness=3
            ),
            color=color.white
        )
        EditorCamera()
        return ursina_app

    def _recalculate_human_model(self, mp_landmarks: HolisticLandmarkerResult) -> None:
        assert(type(self._human_model.model) is Mesh)

        temp = mp_landmarks.pose_landmarks
        world_landmarks = mp_landmarks.pose_world_landmarks
        normal_landmarks = mp_landmarks.pose_landmarks

        self._human_model.model.vertices = []
        self._human_model.model.colors = []
        i = 0
        for landmark in world_landmarks:
            self._human_model.model.vertices.append(
                Vec3(
                    landmark.x,
                    landmark.y,
                    landmark.z
                ))
            new_color = color.blue if normal_landmarks[i].z > 0 else color.red
            self._human_model.model.colors.append(new_color)
            i += 1
        self._human_model.model.triangles = self._land_mark_recognizer.get_pose_connections()
        self._human_model.model.generate()



    def _init_camera(self) -> cv2.VideoCapture:
        video_capture = cv2.VideoCapture(
                                    self.camera_address,
                                    cv2.CAP_FFMPEG,
                                    [
                                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000
                                    ])
        if video_capture.isOpened():
            print("Video capture connection is working")
        else:
            print("Video capture connection failed")
            raise RuntimeError("Video capture connection failed")
        return video_capture

    def __delete__(self, instance):
        self._video_cap.release()


    def _stepFrame(self):
        ret, frame = self._video_cap.read()

        if ret == False:
            print("Couldn't get frame")
        else:
            if self._result_processed:
                self._process_frame(frame)
                if self._result_image is not None:
                    cv2.imshow("MediaPipe" , self._result_image)


        self._ursina_app.step()

    def _process_frame(self, frame: ndarray) -> None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                    )
            self._land_mark_recognizer.detect_async(image, int(time.time() * 1000))
            self._result_processed = False

    def _process_mp_result(self,
                            result: HolisticLandmarkerResult,
                            output_image: mp.Image,
                            timestamp_ms: int) -> None:
        image = self._land_mark_recognizer.add_landmarks_to_image(output_image, result)
        self._result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._result_processed = True
        self._recalculate_human_model(result)
        pass

    def run(self):
        i = 0
        while(True):
            i += 1
            start_time = time.time()
            # process frame
            self._stepFrame()
            # sleep to match the framerate
            end_time = time.time()
            delta_time = (end_time - start_time)
            sleep_time = self.frame_time - delta_time if self.frame_time - delta_time > 0 else 0
            time.sleep(sleep_time)
            # print(f"Frame {i}")
            # print(f"Current max frame rate {1 / delta_time}")
