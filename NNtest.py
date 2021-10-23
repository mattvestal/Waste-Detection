import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2
import globals as g
import matplotlib.image as mpimg

model = tf.keras.models.load_model('modelsBIG/model_0.898.h5') #WILL RAISE WARNING

coords = np.loadtxt('CoordsROI.txt')


classes = []
for i in range(len(coords)):
    test_image = image.load_img('boxoutput/box_' + str(i) + '.jpg',
                                target_size=(100,100))

    # convert image to numpy array
    images = image.img_to_array(test_image)
    # expand dimension of image
    images = np.expand_dims(images, axis=0)
    # making prediction with model
    prediction = model.predict(images)
    prediction = np.argmax(prediction)
    #print(prediction)


    if prediction == 0: #I think this is right
        classes.append(0)
    if prediction == 1:
        classes.append(1)
    #else:
    #    classes.append(2)




#print(classes)
#dir = 'testimages/'
#im1 = mpimg.imread(dir+'food2.jpg')
im1 = mpimg.imread(g.path)
img = im1[:, :, [2, 1, 0]]
result = img.copy()

for i in range(len(coords)):
    testCoord = coords[i]
    x = int(testCoord[0])
    y = int(testCoord[1])
    w = int(testCoord[2])
    h = int(testCoord[3])

    if classes[i] == 0:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    if classes[i] == 1:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #elif classes[i] == 2:
    #    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imwrite('boxesClass.jpg',result)

# show thresh and result
cv2.imshow("bounding_box", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
