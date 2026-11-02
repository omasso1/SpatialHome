import cv2

class Camera(cv2.VideoCapture):
    def __init__(self, address: str):
        self.camera_address = address;
        self.video_capture = cv2.VideoCapture(
            self.camera_address,
            cv2.CAP_ANY, #cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000
            ]
        )
        if self.video_capture.isOpened():
            print("Video capture connection is working")
        else:
            print("Video capture connection failed")
            raise RuntimeError("Video capture connection failed")