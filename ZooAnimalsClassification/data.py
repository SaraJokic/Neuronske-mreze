import keras
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_data(dataframe, img_path, img_size):
    images = []
    labels = []
    boxes = [] #TODO: pitati saru sta sa ovim da radim:(
    for index, row in dataframe.iterrows():
        image_path = os.path.join(img_path, row['filename'])
        img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #TODO: remove ???
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(row['class'])
        boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize images
    labels = np.array(labels)
    boxes = np.array(boxes)
    return images, labels, boxes


class_map = {'eagle': 0, 'panda': 1, 'polar-bear': 2}

# IF DATA IS NOT SPLIT
df = pd.read_csv('dataset/_annotations.csv')
df['class'] = df['class'].map(class_map)
df = shuffle(df).reset_index(drop=True)
temp_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
train_df, val_df = train_test_split(temp_df, test_size=0.2, stratify=temp_df['class'], random_state=42)

image_size = (224, 224)
train_imgs, train_labels, train_boxes = load_data(train_df, 'dataset', image_size)
test_imgs, test_labels, test_boxes = load_data(test_df, 'dataset', image_size)
val_imgs, val_labels, val_boxes = load_data(val_df, 'dataset', image_size)

# Total number of images in each set
print(f"Number of images in training set: {len(train_df)}")
print(f"Number of images in validation set: {len(val_df)}")
print(f"Number of images in test set: {len(test_df)}")

# Number of images in each set (by classes)
print("\nClass distribution in training set:")
print(train_df['class'].value_counts())

print("\nClass distribution in validation set:")
print(val_df['class'].value_counts())

print("\nClass distribution in test set:")
print(test_df['class'].value_counts())

# Save for training (for google colab) TODO:in directory
np.save('train_imgs.npy', train_imgs)
np.save('train_labels.npy', train_labels)
np.save('test_imgs.npy', test_imgs)
np.save('test_labels.npy', test_labels)
np.save('val_imgs.npy', val_imgs)
np.save('val_labels.npy', val_labels)