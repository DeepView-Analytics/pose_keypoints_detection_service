import json
import math
import os
from typing import Optional, Tuple
import onnxruntime as ort
import cv2
import numpy as np
import asyncio
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from v3.bbox import BBox

class RTMPose:
    def __init__(self):
        self.model_path = os.getenv('MODEL_PATH', 'pose_keypoints_estimator_service\model\rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.onnx')
        self.session = ort.InferenceSession(self.model_path)
        self.executor = ThreadPoolExecutor()
        self.min_distance = os.getenv('AUTH_DISTANCE',10)
        self.confidence = os.getenv("CONFIDENCE",0.1)
    def facial_authentication(self,
        left_eye: Tuple[float, float], 
        right_eye: Tuple[float, float], 
        nose: Tuple[float, float], 
        min_distance: float = 10
    ) -> Tuple[str, str]:
        """
        Perform facial authentication and estimate the face bounding box using dynamic scaling.
        """
        # Calculate distances
        min_distance = float(self.min_distance)
        eye_distance = math.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
        print("passed by here")
        if eye_distance < min_distance:
            print(eye_distance)
            print("the distance is good")
            return "False", json.dumps({})

        nose_to_eye_avg = (
            math.sqrt((nose[0] - left_eye[0]) ** 2 + (nose[1] - left_eye[1]) ** 2) +
            math.sqrt((nose[0] - right_eye[0]) ** 2 + (nose[1] - right_eye[1]) ** 2)
        ) / 2

        # Calculate centroid
        centroid_x = (left_eye[0] + right_eye[0] + nose[0]) / 3
        centroid_y = (left_eye[1] + right_eye[1] + nose[1]) / 3

        # Dynamically scale bbox size
        bbox_size = (eye_distance + nose_to_eye_avg) * 2

        # Calculate bbox centered around centroid
        xmin = centroid_x - bbox_size / 2 - 35
        ymin = centroid_y - bbox_size / 2
        xmax = centroid_x + bbox_size / 2 - 35
        ymax = centroid_y + bbox_size / 2
        bbox = BBox(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,conf=-1)
        bbox = bbox.model_dump()
        bbox = json.dumps(bbox)
        return "True", bbox

        

    async def get_simcc_maximum(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = 1
        K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)
        vals = 0.5 * (max_val_x + max_val_y)
        locs[vals <= 0.] = -1
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)
        return locs, vals

    async def preprocess_image(self, image):
        size = image.size
        if image is None:
            raise ValueError(f"Error: Image file could not be decoded.")
        image = np.array(image)
        input_image = cv2.resize(image, (192,256))
        input_image = input_image.astype(np.float32)
        input_image = input_image / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        return input_image, size

    async def postprocess(self, outputs, center, scale, simcc_split_ratio=2.0):
        simcc_x, simcc_y = outputs
        locs, scores = await self.get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio
        keypoints = keypoints / np.array([192,256]) * np.array(scale)
        keypoints = keypoints + np.array(center) - np.array(scale) / 2
        return keypoints, scores

    async def keypoints_to_vector(self, keypoints, num_keypoints=133):
        """
        Converts a list of (id, x, y, confidence) tuples to a fixed-length vector.

        Args:
            keypoints (list): List of tuples (id, x, y, confidence).
            num_keypoints (int): Total number of keypoints (N).

        Returns:
            numpy.ndarray: A fixed-length vector of shape (N * 3,).
        """
        # Initialize vector with NaN for missing values
        vector = np.full(num_keypoints * 3, np.float32(-1))

        for kp in keypoints:
            j, x, y, confidence = kp
            if confidence > float(self.confidence):
                vector[j * 3] = x
                vector[j * 3 + 1] = y
                vector[j * 3 + 2] = confidence

        return vector

    async def run_inference(self, images, bboxs):
        bboxs = list(chain.from_iterable(bboxs))

        input_images = []
        shapes = []

        # Preprocess input images
        for image in images:
            input_image, shape = await self.preprocess_image(image)
            input_images.append(input_image)
            shapes.append(shape)

        input_images = np.stack(input_images, axis=0)

        input_name = self.session.get_inputs()[0].name

        # Run inference in a thread pool
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(self.executor, self.session.run, None, {input_name: input_images})

        # List to store detected poses
        vectors = []
        flags = []
        face_bboxs = []

        # Iterate through outputs to extract keypoints and scores
        for i in range(len(outputs[0])):
            simcc_x, simcc_y = outputs[0][i], outputs[1][i]
            center = (128,96)
            scale = (256,192)

            # Postprocess keypoints and scores
            keypoints, scores = await self.postprocess((simcc_x, simcc_y), center, scale)

            height, width = shapes[i]

            # List to store keypoints for the current image
            keypoint_list = []
            # Process each keypoint
            left_eye = None
            right_eye = None
            nose = None
            for j, keypoint in enumerate(keypoints[0]):
                
                confidence = scores[0][j]
                x = keypoint[0] * height  / 256  + bboxs[i].xmin +30
                y = keypoint[1] * width / 192 + bboxs[i].ymin 
                keypoint_data = (j, x, y, confidence)
                keypoint_list.append(keypoint_data)
                if confidence > 0.4 :
                    if j == 33  : 
                        left_eye = (float(x),float(y))
                    if j == 36  :
                        right_eye = (float(x),float(y))
                    if j == 0  :
                        nose = (float(x),float(y))
            print(left_eye)
            if left_eye is not None and right_eye is not None and nose is not None:
                authFlag , face_bbox =self.facial_authentication(left_eye,right_eye,nose) 
            else:
                authFlag = "False" 
                face_bbox = json.dumps({})
            vector = await self.keypoints_to_vector(keypoint_list)
            flags.append(authFlag)
            face_bboxs.append(face_bbox)
            vectors.append(vector)
        return vectors , flags , face_bboxs
