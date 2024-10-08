
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load imgs, labels and bounding boxes
def load_data(df, img_path, img_size):
    images = []
    labels = []
    boxes = []
    for index, row in df.iterrows():
        image_path = os.path.join(img_path, row['filename'])
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}")
            continue

        img_crop = img[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
        if img_crop.size == 0:
            print(f"Empty crop for {image_path}")
            continue

        img = cv2.resize(img_crop, img_size)
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

# Shuffle the dataframe ?? if needed
train_df = shuffle(train_df).reset_index(drop=True)
test_df = shuffle(test_df).reset_index(drop=True)
val_df = shuffle(val_df).reset_index(drop=True)

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

# Save for training (for google colab)
np.save('train_imgs.npy', train_imgs)
np.save('train_labels.npy', train_labels)
np.save('test_imgs.npy', test_imgs)
np.save('test_labels.npy', test_labels)
np.save('val_imgs.npy', val_imgs)
np.save('val_labels.npy', val_labels)

def plot_class_distribution(df, title):
    class_counts = df['class'].value_counts()
    total_images = class_counts.sum()
    print(f'Total images in {title}: {total_images}')
    for class_name, count in class_counts.items():
        print(f'Class: {class_name}, Number of Images: {count}')

    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(ticks=range(len(class_counts.index)),
               labels=[k for k, v in class_map.items() if v in class_counts.index], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# Plot class distribution for training, validation, and test sets
plot_class_distribution(train_df, 'Class Distribution in Training Set')
plot_class_distribution(val_df, 'Class Distribution in Validation Set')
plot_class_distribution(test_df, 'Class Distribution in Test Set')

# Combined data
combined_df = pd.concat([train_df, val_df, test_df])

# Plot total class distribution
plot_class_distribution(combined_df, 'Total Class Distribution (Train + Validation + Test)')
