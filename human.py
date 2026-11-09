from ursina import Entity, Mesh, color
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarkerResult
import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import numpy as np
from LandMarkRecognizer import LandMarkRecognizer, FaceLandmark, landmark_to_vec3, PoseLandmark, HandLandMarks
import cv2
from camera import Camera
from ursina import Vec3, Vec2
import panda3d.core
from enum import IntEnum, auto
from math_util import norm
import global_timer
import graph_util
from collections import deque

class HumanLandmark(IntEnum):
    # Pose
    POSE_NOSE = 0
    POSE_LEFT_EYE_INNER = auto()
    POSE_LEFT_EYE = auto()
    POSE_LEFT_EYE_OUTER = auto()
    POSE_RIGHT_EYE_INNER = auto()
    POSE_RIGHT_EYE = auto()
    POSE_RIGHT_EYE_OUTER = auto()
    POSE_LEFT_EAR = auto()
    POSE_RIGHT_EAR = auto()
    POSE_MOUTH_LEFT = auto()
    POSE_MOUTH_RIGHT = auto()
    POSE_LEFT_SHOULDER = auto()
    POSE_MIDDLE_SHOULDER = auto()
    POSE_RIGHT_SHOULDER = auto()
    POSE_LEFT_ELBOW = auto()
    POSE_RIGHT_ELBOW = auto()
    POSE_LEFT_WRIST = auto()
    POSE_RIGHT_WRIST = auto()
    POSE_LEFT_PINKY = auto()
    POSE_RIGHT_PINKY = auto()
    POSE_LEFT_INDEX = auto()
    POSE_RIGHT_INDEX = auto()
    POSE_LEFT_THUMB = auto()
    POSE_RIGHT_THUMB = auto()
    POSE_LEFT_HIP = auto()
    POSE_RIGHT_HIP = auto()
    POSE_LEFT_KNEE = auto()
    POSE_RIGHT_KNEE = auto()
    POSE_LEFT_ANKLE = auto()
    POSE_RIGHT_ANKLE = auto()
    POSE_LEFT_HEEL = auto()
    POSE_RIGHT_HEEL = auto()
    POSE_LEFT_FOOT_INDEX = auto()
    POSE_RIGHT_FOOT_INDEX = auto()

    # Face
    FACE_NOSE_TIP = auto()
    FACE_NOSE_TIP_ALT = auto()
    FACE_LEFT_EYE_OUTER_CORNER = auto()
    FACE_LEFT_EYE_INNER_CORNER = auto()
    FACE_RIGHT_EYE_INNER_CORNER = auto()
    FACE_RIGHT_EYE_OUTER_CORNER = auto()
    FACE_LEFT_IRIS_CENTER = auto()
    FACE_RIGHT_IRIS_CENTER = auto()
    FACE_MOUTH_LEFT_CORNER = auto()
    FACE_MOUTH_RIGHT_CORNER = auto()
    FACE_UPPER_LIP_CENTER = auto()
    FACE_LOWER_LIP_CENTER = auto()
    FACE_CHIN = auto()

    # Left hand
    LEFT_HAND_WRIST = auto()
    LEFT_HAND_THUMB_0 = auto()
    LEFT_HAND_THUMB_1 = auto()
    LEFT_HAND_THUMB_2 = auto()
    LEFT_HAND_THUMB_3 = auto()
    LEFT_HAND_INDEX_0 = auto()
    LEFT_HAND_INDEX_1 = auto()
    LEFT_HAND_INDEX_2 = auto()
    LEFT_HAND_INDEX_3 = auto()
    LEFT_HAND_MIDDLE_0 = auto()
    LEFT_HAND_MIDDLE_1 = auto()
    LEFT_HAND_MIDDLE_2 = auto()
    LEFT_HAND_MIDDLE_3 = auto()
    LEFT_HAND_RING_0 = auto()
    LEFT_HAND_RING_1 = auto()
    LEFT_HAND_RING_2 = auto()
    LEFT_HAND_RING_3 = auto()
    LEFT_HAND_PINKY_0 = auto()
    LEFT_HAND_PINKY_1 = auto()
    LEFT_HAND_PINKY_2 = auto()
    LEFT_HAND_PINKY_3 = auto()

    # Right hand
    RIGHT_HAND_WRIST = auto()
    RIGHT_HAND_THUMB_0 = auto()
    RIGHT_HAND_THUMB_1 = auto()
    RIGHT_HAND_THUMB_2 = auto()
    RIGHT_HAND_THUMB_3 = auto()
    RIGHT_HAND_INDEX_0 = auto()
    RIGHT_HAND_INDEX_1 = auto()
    RIGHT_HAND_INDEX_2 = auto()
    RIGHT_HAND_INDEX_3 = auto()
    RIGHT_HAND_MIDDLE_0 = auto()
    RIGHT_HAND_MIDDLE_1 = auto()
    RIGHT_HAND_MIDDLE_2 = auto()
    RIGHT_HAND_MIDDLE_3 = auto()
    RIGHT_HAND_RING_0 = auto()
    RIGHT_HAND_RING_1 = auto()
    RIGHT_HAND_RING_2 = auto()
    RIGHT_HAND_RING_3 = auto()
    RIGHT_HAND_PINKY_0 = auto()
    RIGHT_HAND_PINKY_1 = auto()
    RIGHT_HAND_PINKY_2 = auto()
    RIGHT_HAND_PINKY_3 = auto()

def to_human(
    landmark: PoseLandmark | FaceLandmark | HandLandMarks,
    left_hand: bool | None = None
) -> HumanLandmark:

    if type(landmark) is PoseLandmark:
        prefix = "POSE_"
    elif type(landmark) is FaceLandmark:
        prefix = "FACE_"
    elif type(landmark) is HandLandMarks:
        if left_hand is None:
            raise ValueError("Specify left_hand")
        prefix = "LEFT_HAND_" if left_hand else "RIGHT_HAND_"
    else:
        raise TypeError("Unsupported landmark type")

    return HumanLandmark[prefix + landmark.name]

def from_human(
    landmark: HumanLandmark
) -> tuple[PoseLandmark | FaceLandmark | HandLandMarks, bool | None]:

    name = landmark.name

    if name.startswith("POSE_"):
        return PoseLandmark[name.removeprefix("POSE_")], None

    if name.startswith("FACE_"):
        return FaceLandmark[name.removeprefix("FACE_")], None

    if name.startswith("LEFT_HAND_"):
        return HandLandMarks[name.removeprefix("LEFT_HAND_")], True

    if name.startswith("RIGHT_HAND_"):
        return HandLandMarks[name.removeprefix("RIGHT_HAND_")], False

    raise ValueError("Unknown landmark")

def get_landmark(humanLandmark: HumanLandmark, landmark_result: HolisticLandmarkerResult) -> NormalizedLandmark:
    name = humanLandmark.name

    if name.startswith("POSE_"):
        landmark_enum = PoseLandmark[name.removeprefix("POSE_")]
        return landmark_result.pose_landmarks[landmark_enum]

    if name.startswith("FACE_"):
        landmark_enum = FaceLandmark[name.removeprefix("FACE_")]
        return landmark_result.face_landmarks[landmark_enum]

    if name.startswith("LEFT_HAND_"):
        landmark_enum = HandLandMarks[name.removeprefix("LEFT_HAND_")]
        return landmark_result.left_hand_landmarks[landmark_enum]

    if name.startswith("RIGHT_HAND_"):
        landmark_enum = HandLandMarks[name.removeprefix("RIGHT_HAND_")]
        return landmark_result.right_hand_landmarks[landmark_enum]
    raise ValueError("Unknown landmark")


class Cycle_buffer:
    def __init__(self, size):
        self.array = [0 for i in range(size)]
        self.index = 0

    def add(self, value):
        self.array[self.index] = value
        self.index = (self.index  + 1) % len(self.array)

    def median(self):
        temp = self.array.copy()
        temp.sort()
        return temp[len(self.array) // 2]

class Bone:
    def __init__(self, start: HumanLandmark, end: HumanLandmark, length_cm: int) -> None:
        self.start = start
        self.end = end
        self.length_cm = length_cm

class Joint():
    def __init__(self, landmark: HumanLandmark):
        self.landmark: HumanLandmark = landmark
        self.connections: list[Bone] = []
        self.position: Vec3 = Vec3()
        self.relative_position: Vec3 = Vec3()
        self.rel_velocity: Vec3 = Vec3()
        self.reference_joint: Joint = None
        self.landmark_position = Vec3()
        self.landmark_x_history = Cycle_buffer(5)
        self.landmark_y_history = Cycle_buffer(5)

    def calculate_position(self,
                           parent_bone: Bone,
                           parent_joint: "Joint",
                           landmark_result: HolisticLandmarkerResult,
                           camera: Camera,
                           image: mp.Image) -> bool:
        width = image.width
        height = image.height
        landmark = get_landmark(self.landmark, landmark_result)
        if landmark.x is None or landmark.y is None:
            return False
        new_landmark_position = landmark_to_vec3(landmark)
        self.landmark_x_history.add(new_landmark_position.x)
        self.landmark_y_history.add(new_landmark_position.y)

        coord = Vec2(self.landmark_x_history.median() * width,
                     self.landmark_y_history.median() * height)
        solutions = camera.solve_3d_positions_from_pixel(parent_joint.position,
                                                         coord,
                                                         parent_bone.length_cm)
        min: float = 10000 ** 2
        min_solution = solutions[0]
        for solution in solutions:
            dist = solution.x ** 2 + solution.y ** 2 + solution.z ** 2
            if dist < min:
                min = dist
                min_solution = solution

        prev_rel_position = self.relative_position
        self.position = min_solution
        self.relative_position = self.position - self.reference_joint.position
        pos_diff = (self.relative_position - prev_rel_position)
        self.rel_velocity = pos_diff / (global_timer.delta_time / 1000)

        return True

class Human:
    def __init__(self):
        self._human_model = Entity(
            model=Mesh(
                vertices=[],
                mode='line',
                thickness=3
            ),
            color=color.white,
        )

        self.face_points = np.array([
                    [0.0 , 0.0, 0.0],  # left eye
                    [6.5, 0.0, 0.0],   # right eye
                    [3.25, 5.0, 1.0],  # nose
                    [3.25, 13.0, 0.0]  # chin
                ], dtype=np.float64)

        self.bones = self._init_bones()
        self.skeleton = self._init_skeleton()
        self._add_reference_point_for_joints()
        self.model_verticies, self.model_lines = self._init_model_connections()
        graph_util.add_plot("posX", 
                            lambda:  (global_timer.previous_update_time, self.skeleton[HumanLandmark.POSE_LEFT_WRIST].relative_position.x))
        graph_util.add_plot("posY", 
                            lambda:  (global_timer.previous_update_time, self.skeleton[HumanLandmark.POSE_LEFT_WRIST].relative_position.y))
        graph_util.add_plot("left wrist pos z", 
                             lambda:  (global_timer.previous_update_time, self.skeleton[HumanLandmark.POSE_LEFT_WRIST].relative_position.z))
        graph_util.add_plot("shoulder z", 
                                     lambda:  (global_timer.previous_update_time, self.skeleton[HumanLandmark.POSE_MIDDLE_SHOULDER].position.z))
                

    def _init_skeleton(self) -> dict[HumanLandmark, Joint]:
        graph:dict[HumanLandmark, Joint] = {}
        for bone in self.bones:
            if not bone.start in graph.keys():
                graph[bone.start] = Joint(bone.start)
            graph[bone.start].connections.append(bone)

            if not bone.end in graph.keys():
                graph[bone.end] = Joint(bone.end)
            graph[bone.end].connections.append(bone)

        return graph

    def _init_model_connections(self):
        verticies: list[HumanLandmark] = []
        lines: list[tuple[int, int]] = []

        joint_to_vertices_map: dict[HumanLandmark, int] = {}
        for bone in self.bones:
            start_joint = self.skeleton[bone.start]
            end_joint = self.skeleton[bone.end]

            if start_joint.landmark not in joint_to_vertices_map:
                s = len(joint_to_vertices_map)
                joint_to_vertices_map[start_joint.landmark] = s
                verticies.append(start_joint.landmark)

            if end_joint.landmark not in joint_to_vertices_map:
                s = len(joint_to_vertices_map)
                joint_to_vertices_map[end_joint.landmark] = s
                verticies.append(end_joint.landmark)

            line_start = joint_to_vertices_map[start_joint.landmark]
            line_end = joint_to_vertices_map[end_joint.landmark]
            lines.append((line_start, line_end))

        return verticies, lines
    
    def _add_reference_point_for_joints(self):
        for joint in self.skeleton.values():
            joint.reference_joint = self.skeleton[HumanLandmark.POSE_MIDDLE_SHOULDER]

    def calculate_joints_position(self,
                                  landmark: HumanLandmark,
                                  visited: set[HumanLandmark],
                                  camera: Camera,
                                  landmark_result: HolisticLandmarkerResult,
                                  image: mp.Image) -> None:
        queue = []
        # ret_sequence:list[Joint] = []
        visited.add(landmark)
        # ret_sequence.append(self.skeleton[landmark])
        current_joint = self.skeleton[landmark]
        for bone in current_joint.connections:
            if not bone.end in visited:
                b = self.skeleton[bone.end]
                b.calculate_position(bone, current_joint, landmark_result, camera, image)
                queue.append(bone.end)
                visited.add(HumanLandmark(bone.end))

        for item in queue:
            self.calculate_joints_position(item, visited, camera, landmark_result, image)


    def _init_bones(self) -> list[Bone]:
        bones: list[Bone] = [
            # Face
            Bone(HumanLandmark.POSE_NOSE, HumanLandmark.POSE_LEFT_EYE_INNER, 3),
            Bone(HumanLandmark.POSE_LEFT_EYE_INNER, HumanLandmark.POSE_LEFT_EYE, 1),
            Bone(HumanLandmark.POSE_LEFT_EYE, HumanLandmark.POSE_LEFT_EYE_OUTER, 1),
            Bone(HumanLandmark.POSE_LEFT_EYE_OUTER, HumanLandmark.POSE_LEFT_EAR, 8),
            Bone(HumanLandmark.POSE_NOSE, HumanLandmark.POSE_RIGHT_EYE_INNER, 3),
            Bone(HumanLandmark.POSE_RIGHT_EYE_INNER, HumanLandmark.POSE_RIGHT_EYE, 1),
            Bone(HumanLandmark.POSE_RIGHT_EYE, HumanLandmark.POSE_RIGHT_EYE_OUTER, 1),
            Bone(HumanLandmark.POSE_RIGHT_EYE_OUTER, HumanLandmark.POSE_RIGHT_EAR, 8),
            Bone(HumanLandmark.POSE_MOUTH_LEFT, HumanLandmark.POSE_MOUTH_RIGHT, 5),

            # Shoulders
            # Bone(HumanLandmark.POSE_LEFT_SHOULDER, HumanLandmark.POSE_RIGHT_SHOULDER, 39),
            Bone(HumanLandmark.POSE_MIDDLE_SHOULDER, HumanLandmark.POSE_LEFT_SHOULDER, 39 // 2),
            Bone(HumanLandmark.POSE_MIDDLE_SHOULDER, HumanLandmark.POSE_RIGHT_SHOULDER, 39 // 2),

            # Left arm
            Bone(HumanLandmark.POSE_LEFT_SHOULDER, HumanLandmark.POSE_LEFT_ELBOW, 29),
            Bone(HumanLandmark.POSE_LEFT_ELBOW, HumanLandmark.POSE_LEFT_WRIST, 27),
            Bone(HumanLandmark.POSE_LEFT_WRIST, HumanLandmark.POSE_LEFT_PINKY, 9),
            Bone(HumanLandmark.POSE_LEFT_WRIST, HumanLandmark.POSE_LEFT_INDEX, 10),
            Bone(HumanLandmark.POSE_LEFT_WRIST, HumanLandmark.POSE_LEFT_THUMB, 8),
            Bone(HumanLandmark.POSE_LEFT_PINKY, HumanLandmark.POSE_LEFT_INDEX, 5),

            # Right arm
            Bone(HumanLandmark.POSE_RIGHT_SHOULDER, HumanLandmark.POSE_RIGHT_ELBOW, 29),
            Bone(HumanLandmark.POSE_RIGHT_ELBOW, HumanLandmark.POSE_RIGHT_WRIST, 27),
            Bone(HumanLandmark.POSE_RIGHT_WRIST, HumanLandmark.POSE_RIGHT_PINKY, 9),
            Bone(HumanLandmark.POSE_RIGHT_WRIST, HumanLandmark.POSE_RIGHT_INDEX, 10),
            Bone(HumanLandmark.POSE_RIGHT_WRIST, HumanLandmark.POSE_RIGHT_THUMB, 8),
            Bone(HumanLandmark.POSE_RIGHT_PINKY, HumanLandmark.POSE_RIGHT_INDEX, 5),

            # Torso
            Bone(HumanLandmark.POSE_LEFT_SHOULDER, HumanLandmark.POSE_LEFT_HIP, 50),
            Bone(HumanLandmark.POSE_RIGHT_SHOULDER, HumanLandmark.POSE_RIGHT_HIP, 50),
            Bone(HumanLandmark.POSE_LEFT_HIP, HumanLandmark.POSE_RIGHT_HIP, 30),

            # Thighs
            Bone(HumanLandmark.POSE_LEFT_HIP, HumanLandmark.POSE_LEFT_KNEE, 43),
            Bone(HumanLandmark.POSE_RIGHT_HIP, HumanLandmark.POSE_RIGHT_KNEE, 43),

            # Lower legs
            Bone(HumanLandmark.POSE_LEFT_KNEE, HumanLandmark.POSE_LEFT_ANKLE, 43),
            Bone(HumanLandmark.POSE_RIGHT_KNEE, HumanLandmark.POSE_RIGHT_ANKLE, 43),

            # Feet
            Bone(HumanLandmark.POSE_LEFT_ANKLE, HumanLandmark.POSE_LEFT_HEEL, 7),
            Bone(HumanLandmark.POSE_RIGHT_ANKLE, HumanLandmark.POSE_RIGHT_HEEL, 7),
            Bone(HumanLandmark.POSE_LEFT_HEEL, HumanLandmark.POSE_LEFT_FOOT_INDEX, 23),
            Bone(HumanLandmark.POSE_RIGHT_HEEL, HumanLandmark.POSE_RIGHT_FOOT_INDEX, 23),
            Bone(HumanLandmark.POSE_LEFT_ANKLE, HumanLandmark.POSE_LEFT_FOOT_INDEX, 19),
            Bone(HumanLandmark.POSE_RIGHT_ANKLE, HumanLandmark.POSE_RIGHT_FOOT_INDEX, 19),
        ]

        return bones

    def update_model(self,
                     result: HolisticLandmarkerResult,
                     output_image: mp.Image,
                     camera: Camera) -> None:

        self._calculate_model_points(result, output_image, camera)
        self._recalculate_human_model(result)

        # # print(norm(self.skeleton[HumanLandmark.POSE_LEFT_WRIST].rel_velocity))

    def solve_distance_from_face(self,
                                 result: HolisticLandmarkerResult,
                                 image: mp.Image,
                                 camera: Camera) -> tuple[bool, float]:
        if len(result.face_landmarks) <= 361:
            print("no face")
            return False, 0

        width = image.width
        height = image.height

        leftEye = (landmark_to_vec3(result.face_landmarks[FaceLandmark.LEFT_EYE_INNER_CORNER]) +
                    landmark_to_vec3(result.face_landmarks[FaceLandmark.LEFT_EYE_OUTER_CORNER])) / 2
        rightEye = (landmark_to_vec3(result.face_landmarks[FaceLandmark.RIGHT_EYE_INNER_CORNER]) +
                    landmark_to_vec3(result.face_landmarks[FaceLandmark.RIGHT_EYE_OUTER_CORNER])) / 2
        nose = result.face_landmarks[FaceLandmark.NOSE_TIP]
        chin = result.face_landmarks[FaceLandmark.CHIN]

        image_points = np.array([
            [leftEye.x * width, leftEye.y * height],
            [rightEye.x * width, rightEye.y * height],
            [nose.x * width, nose.y * height],
            [chin.x * width, chin.y * height]
        ], dtype=np.float64);

        # count, rvecs, tvecs = cv2.solveP3P(self.face_points,
        #                 image_points,
        #                 camera._calib_data.mtx,
        #                 camera._calib_data.dist,
        #                 flags=cv2.SOLVEPNP_AP3P)

        flag, rvec, tvec = cv2.solvePnP(self.face_points,
                        image_points,
                        camera._calib_data.mtx,
                        camera._calib_data.dist,
                        flags=cv2.SOLVEPNP_AP3P)
        distance: float = float(np.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2))
        # distances = []
        # for tvec in tvecs:
        #     distance = np.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)
        #     distances.extend(distance)
        return True, distance

    def _calculate_model_points(self,
                                result: HolisticLandmarkerResult,
                                image: mp.Image,
                                camera: Camera):
        ret, possible_distance = self.solve_distance_from_face(result, image, camera)
        if not ret:
            return

        solutions = []
        width = image.width
        height = image.height
        left_shoulder = landmark_to_vec3(result.pose_landmarks[PoseLandmark.LEFT_SHOULDER])
        right_shoulder = landmark_to_vec3(result.pose_landmarks[PoseLandmark.RIGHT_SHOULDER])
        middle_shoulder = (left_shoulder + right_shoulder) / 2
        middle_shoulder.x *= image.width
        middle_shoulder.y *= image.height
        left_shoulder.x *= image.width
        left_shoulder.y *= image.height

        middle_shoulder_vec = camera.pixel_to_vector(Vec2(middle_shoulder.x, middle_shoulder.y))
        # print(f"possible distance: {possible_distance}")
        position = middle_shoulder_vec * possible_distance

        self.skeleton[HumanLandmark.POSE_MIDDLE_SHOULDER].position = position
        self.calculate_joints_position(HumanLandmark.POSE_MIDDLE_SHOULDER, set(), camera, result, image)

        # position_offset = tvecs[0]
        # for landmark in result.pose_world_landmarks:
        #     landmark.x += position_offset[0] / 100
        #     landmark.y += position_offset[1] / 100
        #     landmark.z += position_offset[2] / 100


    def _recalculate_human_model(self, mp_landmarks: HolisticLandmarkerResult) -> None:
        assert(type(self._human_model.model) is Mesh)

        temp = mp_landmarks.pose_landmarks
        world_landmarks = mp_landmarks.pose_world_landmarks
        normal_landmarks = mp_landmarks.pose_landmarks

        self._human_model.model.vertices = []
        self._human_model.model.colors = []
        i = 0
        for landmark in self.model_verticies:
            position = self.skeleton[landmark].position / 100
            self._human_model.model.vertices.append(
                Vec3(
                    position.x,
                    -position.y,
                    position.z
                ))
            i += 1
        self._human_model.model.triangles = self.model_lines

        # phone_cam: Entity = self._room.phone_camera
        # rotation = phone_cam.world_quaternion
        temp = panda3d.core.Vec3(1, 0, 0)
        # rotated_vector = rotation.xform(temp)
        # self._human_model.position = phone_cam.position + rotated_vector
        self._human_model.model.generate()