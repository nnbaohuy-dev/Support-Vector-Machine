import cv2 as cv
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

def load_resize_label_image(input_folder, size):
    images = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        img = cv.imread(file_path, 0)
        resized = cv.resize(img, size)
        images.append(resized)
    
    flatten_images = np.array(images).reshape(len(images), -1) / 255

    label  = os.path.basename(input_folder)

    if label == "NORMAL":
        data_label = np.ones(flatten_images.shape[0], dtype=np.int32)
    else:
        data_label = -np.ones(flatten_images.shape[0], dtype=np.int32)
    
    return flatten_images, data_label

def shuffle(X_1, X_2, y_1, y_2):
    X = np.vstack((X_1, X_2))
    y = np.concatenate((y_1, y_2))

    rng = np.random.default_rng(42)

    perm = rng.permutation(X.shape[0])

    X = X[perm]
    y = y[perm]
    return X, y

def data_preprocessing():
    X_train_normal, y_train_normal = load_resize_label_image("data/chest_xray/train/NORMAL", (128, 128))
    X_train_pneumonia, y_train_pneumonia = load_resize_label_image("data/chest_xray/train/PNEUMONIA", (128, 128))
    X_test_normal, y_test_normal = load_resize_label_image("data/chest_xray/test/NORMAL", (128, 128))
    X_test_pneumonia, y_test_pneumonia = load_resize_label_image("data/chest_xray/test/PNEUMONIA", (128, 128))
    X_val_normal, y_val_normal = load_resize_label_image("data/chest_xray/val/NORMAL", (128, 128))
    X_val_pneumonia, y_val_pneumonia = load_resize_label_image("data/chest_xray/val/PNEUMONIA", (128, 128))

    X_train, y_train = shuffle(X_train_normal, X_train_pneumonia, y_train_normal, y_train_pneumonia)
    X_test, y_test = shuffle(X_test_normal, X_test_pneumonia, y_test_normal, y_test_pneumonia)
    X_val, y_val = shuffle(X_val_normal, X_val_pneumonia, y_val_normal, y_val_pneumonia)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

