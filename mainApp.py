from ursina import Ursina, load_blender_scene, EditorCamera, Mesh, Entity, Vec3, color, Vec2, camera
from panda3d.core import Quat
import panda3d.core
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
from GUI import GUI
from camera import Camera

fileDir = Path(__file__).resolve().parent

class MainApp():
    def __init__(self):
        self.camera_address = "http://192.168.1.134:8080/stream"
        self._video_cap = self._init_camera(self.camera_address)
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
        Gui.add_button("new_button")
        return Gui

    def _init_ursina(self) -> Ursina:
        ursina_app =  Ursina()
        self._room = load_blender_scene("room", fileDir / "roomModel", reload=True, vertex_colors=True)

        self._human_model = Entity(
            model=Mesh(
                vertices=[],
                mode='line',
                thickness=3
            ),
            color=color.white,
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
                    -landmark.y,
                    landmark.z
                ))
            new_color = color.blue if normal_landmarks[i].z > 0 else color.red
            self._human_model.model.colors.append(new_color)
            i += 1
        self._human_model.model.triangles = self._land_mark_recognizer.get_pose_connections()
        self._human_model.world_quaternion
        phone_cam: Entity = self._room.phone_camera
        rotation = phone_cam.world_quaternion
        temp = panda3d.core.Vec3(1, 0, 0)
        rotated_vector = rotation.xform(temp)
        self._human_model.position = phone_cam.position + rotated_vector
        self._human_model.model.generate()



    def _init_camera(self, address: str) -> Camera:
        video_capture = Camera(address)
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
