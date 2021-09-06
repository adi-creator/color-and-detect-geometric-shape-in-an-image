import cv2
from matplotlib import pyplot as plt
import Functions as fun
import numpy as np

img = cv2.imread('img2.png')

prob1 = 0.05

noisy_img = fun.sp_noise(img, prob1)

median = fun.reduce_noise(noisy_img, prob1)

result = fun.colorShapes(median)

plt.subplot(131)
plt.title('input')
plt.imshow(img[:, :, ::-1])
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.title('input')
plt.imshow(noisy_img)
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.title('filltered result')
plt.imshow(result)
plt.xticks([])
plt.yticks([])
plt.show()





