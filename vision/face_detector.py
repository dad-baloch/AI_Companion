"""Face Detector — captures frames from camera and detects faces using OpenCV."""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class FaceDetector:
    """Real-time face detection using OpenCV's DNN or Haar cascades."""

    def __init__(self, camera_index: int = 0, use_dnn: bool = True):
        self.camera_index = camera_index
        self.use_dnn = use_dnn
        self.cap = None
        self.face_cascade = None
        self.net = None

    def initialize(self):
        """Initialize the camera and face detection model."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"[Vision] Cannot open camera {self.camera_index}")
            raise RuntimeError("Camera not available.")

        if self.use_dnn:
            # Use OpenCV's DNN face detector (more accurate)
            proto_path = cv2.data.haarcascades + "deploy.prototxt"
            model_path = "models/vision/res10_300x300_ssd_iter_140000.caffemodel"
            try:
                self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                logger.info("[Vision] DNN face detector loaded.")
            except Exception:
                logger.warning("[Vision] DNN model not found, falling back to Haar cascade.")
                self.use_dnn = False

        if not self.use_dnn:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            logger.info("[Vision] Haar cascade face detector loaded.")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            logger.error("[Vision] Camera not initialized.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("[Vision] Failed to capture frame.")
            return None
        return frame

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame.
        
        Args:
            frame: BGR image as numpy array.
            
        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        if self.use_dnn and self.net is not None:
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [tuple(f) for f in faces]

    def get_face_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get the largest face region-of-interest from a frame."""
        faces = self.detect_faces(frame)
        if not faces:
            return None

        # Return the largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        return frame[y:y+h, x:x+w]

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            logger.info("[Vision] Camera released.")
