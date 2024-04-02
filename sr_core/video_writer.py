import cv2 as cv
import os
from os.path import join

input_dir = r'E:\Programming\Python\magnet\logs\MagNetv2_p32_b6_cMSE_SRRb\predictions'
# input_dir = '../video'
images = os.listdir(input_dir)
print(images)
print(len(images))
img_array = []
for image in images:
    path = join(input_dir, image)
    img = cv.imread(path)
    height, width, channels = img.shape
    size = (width, height)
    img_array.append(img)

out = cv.VideoWriter('predictions.avi', cv.VideoWriter_fourcc(*'DIVX'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()