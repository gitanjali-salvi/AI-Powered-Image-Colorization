import os
import cv2
import numpy as np

class Colorizer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
        # Ensure model directory 
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Loading and preparing the model
        self.net = self.load_model()
        self.prepare_model()

    def load_model(self):
        prototxt_path = os.path.join(self.model_dir, 'colorization_deploy_v2.prototxt')
        model_path = os.path.join(self.model_dir, 'colorization_release_v2.caffemodel')
        points_path = os.path.join(self.model_dir, 'pts_in_hull.npy')

        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        pts_in_hull = np.load(points_path)
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype('float32')]
        net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype='float32')]

        return net

    def prepare_model(self):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def colorize(self, image):
        if image.ndim == 2:  # If grayscale (single channel), convert to 3-channel grayscale
           image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        normalized = image.astype('float32') / 255.0
        lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        resized_l = cv2.resize(l_channel, (224, 224))
        resized_l -= 50

        blob = cv2.dnn.blobFromImage(resized_l)
        self.net.setInput(blob)

        ab_output = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_output = cv2.resize(ab_output, (image.shape[1], image.shape[0]))

        lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_output), axis=2)
        rgb_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2RGB)
        rgb_output = np.clip(rgb_output, 0, 1)

        return (rgb_output * 255).astype('uint8')
