import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt

file_name = os.listdir('./image')

output_dir = './output_img/'	        
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

for i in file_name:
    img = cv2.imread('./image/'+i, 1)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    boundaries1 = [([0, 43, 46],[10, 255, 255])]
    for (lower1, upper1) in boundaries1:
        lower1 = np.array(lower1, dtype = "uint8")
        upper1 = np.array(upper1, dtype = "uint8")
        mask = cv2.inRange(img_hsv, lower1, upper1)
        output1 = cv2.bitwise_and(img, img_hsv, mask = mask)

    Y,U,V = cv2.split(img_yuv)
    rusty = cv2.inRange(V,170,255)
    
    cv2.imshow("origin",img)
    cv2.imshow("rusty",rusty)
    cv2.imshow("output1",output1)
    cv2.waitKey(0)