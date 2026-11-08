import cv2
from cv2.typing import MatLike, Size
import numpy as np
from pathlib import Path
import time
from ursina import Audio, Vec2, Vec3
import pickle
import os


fileDir = Path(__file__).resolve().parent

class Camera:
    class Calibration_data:
        def __init__(self,
                     mtx: MatLike | None = None,
                     dist: MatLike | None = None,
                     rvecs = None,
                     tvecs = None) -> None:
            self.mtx:MatLike = mtx
            self.dist:MatLike = dist
            self.rvecs = rvecs
            self.tvecx = tvecs

    def __init__(self, address: str,
                 record_video: bool = False,
                 read_video: bool = False,
                 record_read_path: Path = fileDir / Path("video/vid01.mp4")):

        self.camera_address = address;
        if read_video:
            self.camera_address = record_read_path

        self.video_capture = cv2.VideoCapture(
            self.camera_address,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000
            ]
        )

        self.video_writer = None

        if self.video_capture.isOpened():
            print("Video capture connection is working")
        else:
            print("Video capture connection failed")
            raise RuntimeError("Video capture connection failed")

        self._calib_pictures: list[MatLike] = []
        self._calib_data = self.Calibration_data()
        self._record_video: bool = record_video
        self._read_video: bool = read_video
        self._recorded_video: list[MatLike] = []
        self._record_read_path:Path = record_read_path
        self._video_width:int = 0
        self._video_height:int = 0
        self.load_calib_pictures()
        self.load_calib_data()

    def load_calib_pictures(self) -> None:
        calib_pictures_paths = Path('./images').glob('./*.png')
        for path in calib_pictures_paths:
            img = cv2.imread(path)
            if img is not None:
                self._calib_pictures.append(img)
            else:
                raise RuntimeError(f'Couldn;t load picture {path}')

    def load_calib_data(self) -> None:
        calib_path = fileDir / 'calibData' / 'calib.obj'
        if not os.path.exists(calib_path):
            return

        with open(calib_path, 'br') as file:
            self._calib_data = pickle.load(file)

    def save_calib_data(self) -> None:
        with open(fileDir / 'calibData' / 'calib.obj', 'bw') as file:
            pickle.dump(self._calib_data, file)

    def release(self):
        self.video_capture.release()

    def read(self) -> tuple[bool, MatLike]:
        ret, frame = self.video_capture.read()

        if(ret and self._record_video):
            self._recorded_video.append(frame)

            if self.video_writer is None:
                self._video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_writer = cv2.VideoWriter(self._record_read_path,
                            cv2.VideoWriter.fourcc('M', 'P', '4', 'V'),
                            10.0,
                            (self._video_width, self._video_height),
                            True)

            self.video_writer.write(frame)
        if self._read_video:
            if not ret:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                
        return (ret, frame)

    def take_calib_picture(self):
        for i in range(10):
            start = time.time()
            while(time.time() - start < 5):
                ret, frame = self.read()
                time.sleep(0.016)

            sound = Audio('sine', autoplay=False)
            sound.play()
            ret, frame = self.read()
            if ret == False:
                print("Couldn't get frame")
            else:
                self._calib_pictures.append(frame)

    def process_calib_pictures(self) -> None:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        number_found = 0
        correct_picutres = []
        gray = None
        for picture in self._calib_pictures:
            gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret == True:
                correct_picutres.append(picture)
                img_path = f'images/calib_pic_{number_found}.png'
                ret = cv2.imwrite(img_path, picture)
                if ret == False:
                    raise RuntimeError(f'Failed to save in path {img_path}')

                number_found += 1
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(picture, (7, 7), corners2, ret)
                # cv2.imshow(f'img_{number_found}', picture)

            print(f"number found {number_found}")
        if gray is not None:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            self._calib_data.mtx = mtx
            self._calib_data.dist = dist
            self._calib_data.rvecs = rvecs
            self._calib_data.tvecx = tvecs
            self.save_calib_data()


        self._calib_pictures = correct_picutres

    def save_video(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()

    def pixel_to_vector(self, pixel_coord: Vec2) -> Vec3:
        point = np.array([[[pixel_coord.x, pixel_coord.y]]], dtype=np.float64)

        undistortedPoint = cv2.undistortPoints(point, self._calib_data.mtx, self._calib_data.dist)
        x, y = undistortedPoint[0, 0]
        direction = np.array([x, y, 1.0])
        direction /= np.linalg.norm(direction)
        return Vec3(direction[0], direction[1], direction[2])

    def position_3d_to_pixel(self, position: Vec3) -> Vec2:
        pixel = np.array([[position.x, position.y, position.z]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            pixel,
            np.zeros(3),  # brak dodatkowej rotacji
            np.zeros(3),  # brak dodatkowej translacji
            self._calib_data.mtx,
            self._calib_data.dist
            )
        return Vec2(image_points[0, 0, 0], image_points[0, 0, 1])
    
    def solve_3d_positions_from_pixel(self, point: Vec3, pixel_coord: Vec2, distance: float) -> list[Vec3]:
        direction = self.pixel_to_vector(pixel_coord)
        a = direction.x ** 2 + direction.y ** 2 + direction.z ** 2
        b = -2 * direction.x * point.x - 2 * direction.y * point.y - 2 * direction.z * point.z
        c = point.x ** 2 + point.y ** 2 + point.z ** 2 - distance ** 2
        solutions = np.roots([a, b, c])
        return [direction * solution for solution in solutions]

    # def __del__(self):
    #     if not self._record_video:
    #         return

    #     writer = cv2.VideoWriter(self._record_read_path.name,
    #                              cv2.VideoWriter.fourcc('M', 'P', '4', 'V'),
    #                              10.0,
    #                              (self._video_width, self._video_height),
    #                              True)

        # for img in self._recorded_video:
        #     writer.write(img)