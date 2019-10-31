import cv2
import numpy as np
import os 
#Read the Rust Photograph
file_name = os.listdir('./image')
for i in file_name:
    img = cv2.imread('./image/'+i, 1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Set different boundaries for different shades of rust
    boundaries1 = [ ([58, 57, 101], [76, 95, 162]) ]
    boundaries2 = [ ([26, 61, 111], [81, 144, 202]) ]
    boundaries3 = [ ([44, 102, 167], [115, 169, 210]) ]

    #Highlight out the shades of rust
    for (lower1, upper1) in boundaries1:
        lower1 = np.array(lower1, dtype = "uint8")
        print(lower1)
        upper1 = np.array(upper1, dtype = "uint8")
        print(upper1)
        mask = cv2.inRange(img, lower1, upper1)
        output1 = cv2.bitwise_and(img, img, mask = mask)

    for (lower2, upper2) in boundaries2:
        lower2 = np.array(lower2, dtype = "uint8")
        upper2 = np.array(upper2, dtype = "uint8")
        mask = cv2.inRange(img, lower2, upper2)
        output2 = cv2.bitwise_and(img, img, mask = mask)

    for (lower3, upper3) in boundaries3:
        lower3 = np.array(lower3, dtype = "uint8")
        upper3 = np.array(upper3, dtype = "uint8")
        mask = cv2.inRange(img, lower3, upper3)
        output3 = cv2.bitwise_and(img, img, mask = mask)

    #Combine the 3 different masks with the different shades into 1 image file
    final = cv2.bitwise_or(output1, output2, output3)
    cv2.imshow("original", img)
    cv2.imshow("output1", output1)
    cv2.imshow("output2", output2)
    cv2.imshow("output3", output3)
    cv2.imshow("final", final)
    cv2.waitKey(0)