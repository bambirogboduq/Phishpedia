import imp
import unittest
import os
import cv2
import numpy as np
from src.detectron2_pedia import inference
import time

class TestInference(unittest.TestCase):
    def setUp(self):
        # Set up paths to test data
        self.image_path = 'path/to/test/image.png'
        self.cfg_path = 'path/to/test/config.yaml'
        self.weights_path = 'path/to/test/weights.pth'
        self.conf_threshold = 0.5

        # Ensure the test files exist
        self.assertTrue(os.path.exists(self.image_path))
        self.assertTrue(os.path.exists(self.cfg_path))
        self.assertTrue(os.path.exists(self.weights_path))

        # Initialize predictor
        self.predictor = inference.config_rcnn(self.cfg_path, self.weights_path, self.conf_threshold)

    def test_pred_rcnn_time(self):
        start_time = time.time();
        inference.pred_rcnn(self.image_path, self.predictor)
        end_time = time.time();
        execution_time = end_time - start_time;
        print(f"Execution time of pred_rcnn() is: {execution_time} seconds")
        

    def test_vis(self):
        start_time = time.time();
        image = inference.vis(self.image_path, np.array([[10, 10, 50, 50]]))
        end_time = time.time();
        execution_time = end_time - start_time;
        print(f"Execution time of vis() is: {execution_time} seconds")

if __name__ == '__main__':
    unittest.main()