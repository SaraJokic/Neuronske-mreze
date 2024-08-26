import keras
import numpy as np
import os
from keras.api.models import load_model
from sklearn.metrics import classification_report

import data


def report_model(md, model_name, images, labels, classes):
    print("**********REPORT FOR ", model_name.upper(), " MODEL********** /n")
    predictions = md.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    report = classification_report(labels, predicted_labels, target_names=list(classes.keys()),
                                   labels=list(classes.values()))
    print(report)
    print("******************************************")


test_imgs = np.load('test_imgs.npy')
test_labels = np.load('test_labels.npy')
'''
train_imgs = np.load('train_imgs.npy')
train_labels = np.load('train_labels.npy')
val_imgs = np.load('val_imgs.npy')
val_labels = np.load('val_labels.npy')
train_datagen = ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=True,
                                   rotation_range=20)
val_datagen = ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=True,
                                   rotation_range=20)

train_set = train_datagen.flow(train_imgs,
                                     train_labels,
                                     batch_size=32,
                                     shuffle=True)
val_set = val_datagen.flow(val_imgs,
                                 val_labels,
                                 batch_size=32,
                                 shuffle=True)
'''

train_vgg = True

if train_vgg:
    #VGG MODEL
    if os.path.exists('trained_cropped/vgg_model.h5'):
        model = load_model('trained_cropped/vgg_model.h5')
        print("Model loaded.")
        report_model(model, "vgg", test_imgs, test_labels, data.class_map)
    else:#224
        print("Build vgg model")
        model = keras.Sequential()
        res_net_pre = keras.applications.VGG19(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
        for layer in res_net_pre.layers:
            layer.trainable = False  # weights don't update during training
        # include_top=False: we can add custom layers

        model.add(res_net_pre)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(3, activation='softmax'))
        model.summary()
        model.save("saved_models/pre_trained_vgg.h5")
else:
    #RESNET MODEL
    if os.path.exists('trained_cropped/resnet_model.h5'):
        model = load_model('trained_cropped/resnet_model.h5')
        print("Model loaded.")
        report_model(model, "resnet", test_imgs, test_labels, data.class_map)
    else:#224
        print("Build resnet model")
        model = keras.Sequential()
        res_net_pre = keras.applications.VGG19(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
        for layer in res_net_pre.layers:
            layer.trainable = False  # weights don't update during training
        # include_top=False: we can add custom layers

        model.add(res_net_pre)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(3, activation='softmax'))
        model.summary()
        model.save("saved_models/pre_trained_vgg.h5")
