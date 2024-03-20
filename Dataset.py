import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path, width, height):
        self.path = path
        self.width = width
        self.height = height

    def read_dataset(self):
        images = []
        targets = []

        root = os.listdir(self.path)
        for i in root:
            root_path = os.path.join(self.path, i)
            folders = os.listdir(root_path)
            for folder in folders:
                folder_path = os.path.join(root_path, folder)
                files = os.listdir(folder_path)
                for file in files:
                    if file.endswith(".bmp"):
                        file_path = os.path.join(folder_path, file)
                        image = cv2.imread(file_path)
                        image = cv2.resize(image, dsize=(self.width, self.height))

                        images.append(image)
                        targets.append(int(i))

        images = np.array(images)
        targets = np.array(targets) - 1

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
