import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
from sys import getsizeof

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

orgs = os.path.expanduser('data/DATASET/TEST/O')
organics = glob.glob(orgs + '/*.jpg')
NotOrgs = os.path.expanduser('data/DATASET/TEST/R')
inorganics = glob.glob(NotOrgs + '/*.jpg')

dataOrganic = []
for i in range(len(organics)):
    x = organics[i]
    x = cv2.imread(x)
    x = cv2.resize(x, (100,100), interpolation = cv2.INTER_AREA)
    data = [x,0]
    dataOrganic.append(data)
#dataOrganic = np.array(dataOrganic)

dataInorganic = []
for i in range(len(inorganics)):
    x = inorganics[i]
    x = cv2.imread(x)
    x = cv2.resize(x, (100,100), interpolation = cv2.INTER_AREA)
    data = [x,1]
    dataInorganic.append(data)
#dataInorganic = np.array(dataInorganic)

fullSet = dataOrganic + dataInorganic

orgs = os.path.expanduser('data/DATASET/TRAIN/O')
organics = glob.glob(orgs + '/*.jpg')
NotOrgs = os.path.expanduser('data/DATASET/TRAIN/R')
inorganics = glob.glob(NotOrgs + '/*.jpg')


dataOrganic = []
for i in range(len(organics)):
    x = organics[i]
    x = cv2.imread(x)
    x = cv2.resize(x, (100,100), interpolation = cv2.INTER_AREA)
    data = [x,0]
    dataOrganic.append(data)
#dataOrganic = np.array(dataOrganic)

dataInorganic = []
for i in range(len(inorganics)):
    x = inorganics[i]
    x = cv2.imread(x)
    x = cv2.resize(x, (100,100), interpolation = cv2.INTER_AREA)
    data = [x,1]
    dataInorganic.append(data)
#dataInorganic = np.array(dataInorganic)

fullSetTest = dataOrganic + dataInorganic
print('Check 1')


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in fullSet:
    x_train.append(feature)
    y_train.append(label)

for feature, label in fullSetTest:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

#x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

#x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)
del dataOrganic
del dataInorganic
del fullSetTest
print("Check 2")
#print(getsizeof(x_train), getsizeof(y_train), getsizeof(x_val),getsizeof(y_val))

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

print('Check 3')
###DEFINE MODEL CHANGE CODE HERE
model = Sequential()
model.add(Conv2D(64, 5,padding="same", activation="relu", input_shape=(100,100,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 5, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 5, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))

opt = Adam(lr=0.00001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,
              metrics = ['accuracy'])


n_epochs = 50#Change this too
print("Training...")
switch = 2
if switch == 1:
    history = model.fit(x_train,y_train,epochs = n_epochs , validation_data = (x_val, y_val),callbacks=[
              # Stopping our training if val_accuracy doesn't improve after 20 epochs
              tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                               patience=20),
              # Saving the best weights of our model in the model directory

              # We don't want to save just the weight, but also the model architecture
              tf.keras.callbacks.ModelCheckpoint('modelsBIG/model_{val_accuracy:.3f}.h5',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               monitor='val_accuracy'
                                                 )
        ])
else:
    print(model.summary())

'''
history.history.keys()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
'''

'''
epochs_range = range(5)
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
'''
