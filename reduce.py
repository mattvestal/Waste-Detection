from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import impackage as im
import numpy as np
import cv2
from PIL import Image
#import tensorflow as tf
dir = 'testimages/'
fname = 'food2'
readImage = dir + fname+'.jpg'


mode = 2 #1 to resize, 2 for regular image
if mode == 1:
    im1 = Image.open(readImage)
    test = np.array(im1)
    if test.shape[0] > 1000 and test.shape[1] > 600:
        im1 = im1.resize((1000,600))

    im1 = np.array(im1)
    bthresh = 200

if mode == 2:
    im1 = mpimg.imread(readImage)
    bthresh = 1000


x = im1.shape[0]
y = im1.shape[1]

thresh = 200
l = 0.1 #set low and high thresholds for canny
h = 0.3
gray = im.grayscale(im1,x,y)
binary = im.binarize(thresh,gray,x,y)
dilated = im.fill(9,binary,x,y)

filt = im.canny(h,l,gray,x,y)
filt = im.fill(4,filt,x,y)
#filt2 = im.dilate(12,filt,x,y)
overlay = im.overlay(dilated,filt,x,y)
overlay = im.fill(5,overlay,x,y)
overlay = im.dilate(7,overlay,x,y)
overlay = im.fill(16,overlay,x,y)
#overlay = im.dilate(3,overlay,x,y)
overlay2 = im.overlay3(im1,overlay,x,y)
#filt = im.flip(filt,x,y)
plt.imshow(overlay2)
plt.show()

im_bgr = overlay2[:, :, [2, 1, 0]]
#cv2.imwrite('Overlay.jpg',im_bgr)

plt.imshow(overlay2)
plt.savefig('overlaycolor.jpg')

writeto = cv2.convertScaleAbs(im_bgr, alpha=(255.0))
cv2.imwrite('Overlaycolor.jpg', writeto)

writeto = cv2.convertScaleAbs(filt, alpha=(255.0))
cv2.imwrite('edges.jpg', writeto)

writeto = cv2.convertScaleAbs(dilated, alpha=(255.0))
cv2.imwrite('dilated.jpg', writeto)

writeto = cv2.convertScaleAbs(overlay, alpha=(255.0))
cv2.imwrite('Overlay.jpg', writeto)



'''
fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
fig.suptitle('Canny Edges, Binary, Binary Overlay')
ax1.imshow(filt,'gray')
cv2.imwrite('edges.jpg',filt)
ax2.imshow(dilated,'gray')
cv2.imwrite('dilated.jpg',dilated)
ax3.imshow(overlay,'gray')
cv2.imwrite('overlay.jpg',overlay)
plt.show()
'''

print(overlay.shape)
img = im1[:, :, [2, 1, 0]]

thresh = overlay#cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]
thresh = thresh.astype(np.uint8)
plt.imshow(thresh,'gray')
plt.show()
# get contours
result = img.copy()
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
n = 0
coords = []
for cntr in contours:
    x,y,w,h = cv2.boundingRect(cntr)
    if w*h > bthresh:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #print(n)
        n += 1
        print("x,y,w,h:",x,y,w,h)
        coords.append([x,y,w,h])

print(n)

np.savetxt('Coords'+fname+'.txt', coords)

for i in range(n):
    testCoord = coords[i]
    x = testCoord[0]
    y = testCoord[1]
    w = testCoord[2]
    h = testCoord[3]

    xmin = x
    xmax = x+w
    ymin = y
    ymax = y+h
    roi = img[ymin:ymax,xmin:xmax]#.copy()
    cv2.imwrite("boxoutput/box_{}.jpg".format(str(i)), roi)




# save resulting image
cv2.imwrite('boxes2.jpg',result)

# show thresh and result
cv2.imshow("bounding_box", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
