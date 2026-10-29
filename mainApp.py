from ursina import Ursina, load_blender_scene, EditorCamera
from ursina.shaders import lit_with_shadows_shader
from pathlib import Path
from typing import cast
import time
import cv2
from numpy import ndarray

fileDir = Path(__file__).resolve().parent

class MainApp():
    def __init__(self):
        self.camera_address = "http://192.168.1.134:8080/stream"
        self._video_cap = self._init_camera()
        self._ursina_app = self._init_ursina()
        self.frame_rate = 120
        self.frame_time = 1 / self.frame_rate
        self.show_camera = True

    def _init_ursina(self) -> Ursina:
        ursina_app =  Ursina()
        self._room = load_blender_scene("room", fileDir / "roomModel", reload=True, vertex_colors=True)
        EditorCamera()
        return ursina_app


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
            return
        else:
            self._process_frame(frame)

        self._ursina_app.step()

    def _process_frame(self, frame: ndarray):

        pass

    def run(self):
        while(True):
            start_time = time.time()
            # process frame
            self._stepFrame()
            # sleep to match the framerate
            end_time = time.time()
            delta_time = (end_time - start_time)
            sleep_time = self.frame_time - delta_time if self.frame_time - delta_time > 0 else 0
            time.sleep(sleep_time)
            # print(f"Current max frame rate {1 / delta_time}")
