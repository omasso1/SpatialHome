from ursina import Ursina, load_blender_scene, EditorCamera, Mesh, Entity, Vec3, color, Vec2, camera, application
from pathlib import Path
from typing import cast
import time
import cv2
from LandMarkRecognizer import LandMarkRecognizer, FaceLandmark, landmark_to_vec3, PoseLandmark
from numpy import ndarray
import mediapipe as mp
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarkerResult
from GUI import GUI
from camera import Camera
from human import Human
import global_timer
import graph_util

fileDir = Path(__file__).resolve().parent

class MainApp():
    def __init__(self):
        self.camera_address = "http://192.168.1.151:8080/stream"
        self._video_cap = self._init_camera(self.camera_address)
        self._human: Human = Human()
        self._ursina_app = self._init_ursina()
        self._land_mark_recognizer = LandMarkRecognizer(self._process_mp_result)

        self._Gui = self._init_GUI()

        self.frame_rate = 120
        self.frame_time = 1 / self.frame_rate
        self._show_camera = True
        self._result_processed = True
        self._result_image = None

    def _init_GUI(self) -> GUI:
        Gui = GUI()
        config = Gui.Slider_config(-180,
                          180,
                          0,
                          Vec2(0.05,-0.05),
                          1,
                          lambda: setattr(
                            self._room.phone_camera, 'rotation_x', self._Gui.sliders["rotation_x"].value
                          ))
        Gui.add_slider("rotation_x", config)

        config = Gui.Slider_config(-180,
                          180,
                          0,
                          Vec2(0.05,-0.10),
                          1,
                          lambda: setattr(
                            self._room.phone_camera, 'rotation_y', self._Gui.sliders["rotation_y"].value
                          ))
        Gui.add_slider("rotation_y", config)

        config = Gui.Slider_config(-180,
                          180,
                          0,
                          Vec2(0.05,-0.15),
                          1,
                          lambda: setattr(
                            self._room.phone_camera, 'rotation_z', self._Gui.sliders["rotation_z"].value
                          ))
        Gui.add_slider("rotation_z", config)

        config = Gui.Button_config(
            "Take  calibration picutres",
            Vec2(0.15, -0.2),
            self._video_cap.take_calib_picture
        )
        Gui.add_button("Take calibration picutre", config)

        config = Gui.Button_config(
            "Calibrate camera",
            Vec2(0.15, -0.3),
            self._video_cap.process_calib_pictures
        )
        Gui.add_button("Take calibration picutre", config)
        return Gui

    def _init_ursina(self) -> Ursina:
        ursina_app =  Ursina()
        self._room = load_blender_scene("room", fileDir / "roomModel", reload=True, vertex_colors=True)

        self.human = Human()

        EditorCamera()

        ursina_app.win.setCloseRequestEvent('close-request')
        ursina_app.accept('close-request', self.on_close)

        return ursina_app


    def _init_camera(self, address: str) -> Camera:
        video_capture = Camera(address, read_video=False, record_video=False)
        return video_capture

    # def __del__(self):
    #     self._video_cap.release()


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
            self._land_mark_recognizer.detect_async(image, global_timer.previous_update_time)
            self._result_processed = False

    def _process_mp_result(self,
                            result: HolisticLandmarkerResult,
                            output_image: mp.Image,
                            timestamp_ms: int) -> None:

        image = self._land_mark_recognizer.add_landmarks_to_image(output_image, result)
        self._result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._result_processed = True
        # elbow_z = result.pose_world_landmarks[PoseLandmark.LEFT_SHOULDER].z
        # shoulder_z = result.pose_world_landmarks[PoseLandmark.RIGHT_SHOULDER].z
        # print(f"distance {elbow_z - shoulder_z}")
        self._human.update_model(result, output_image, self._video_cap) # self._recalculate_human_model(result)
        # self._calculate_model_points(result, output_image)


    def run(self):
        i = 0
        while(True):
            i += 1
            global_timer.update()
            start_time = time.time()
            # process frame
            self._stepFrame()
            # sleep to match the framerate
            end_time = time.time()
            step_time = (end_time - start_time)
            sleep_time = self.frame_time - step_time if self.frame_time - step_time > 0 else 0
            if i % 5 == 0:
                graph_util.update_plots()
            time.sleep(sleep_time)

            # print(f"Frame {i}")
            # print(f"Current max frame rate {1 / step_time}")

    def on_close(self):
        self._video_cap.save_video()
        graph_util.close()
        application.quit()


