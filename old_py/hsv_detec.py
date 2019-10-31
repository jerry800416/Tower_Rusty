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
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Set different boundaries for different shades of rust
    boundaries1 = [([0, 43, 46],[10, 255, 255])]
    # boundaries2 = [([0, 0, 0], [180, 200, 20])]

    #detec all range red 
    for (lower1, upper1) in boundaries1:
        lower1 = np.array(lower1, dtype = "uint8")
        upper1 = np.array(upper1, dtype = "uint8")
        mask = cv2.inRange(img_hsv, lower1, upper1)
        output1 = cv2.bitwise_and(img, img_hsv, mask = mask)
        # output1 = cv2.cvtColor(output1, cv2.COLOR_HSV2BGR)
    
    #detec all range black
    # for (lower2, upper2) in boundaries2:
    #     lower2 = np.array(lower2, dtype = "uint8")
    #     upper2 = np.array(upper2, dtype = "uint8")
    #     mask = cv2.inRange(img_hsv, lower2, upper2)
    #     output2 = cv2.bitwise_and(img, img_hsv, mask = mask)
    #     output2 = cv2.cvtColor(output2, cv2.COLOR_HSV2BGR)
    # final = cv2.bitwise_or(output1, output2)


    # cv2.imshow("original", img)
    # cv2.imshow("output1", output1)
    # cv2.imshow("output2", output2)
    # cv2.imshow("final", final)
    # cv2.waitKey(0)

    # result = np.concatenate([img, output1], axis=1)   #橫向拼接
    result = cv2.addWeighted(img, 0.2, output1, 0.8, 0)  #融合圖片
    result = np.concatenate([img, result], axis=1)
    #將拼接好的檔案丟到分類好的路徑資料夾下
    cv2.imwrite(output_dir+i,result)