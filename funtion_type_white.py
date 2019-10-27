import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance

def single_np(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size # 計算單個元素出現次數


def rust_identification (input_img ,output_img = 'no need'):
    '''
    input_img : The filename of the input image 
    output_img : The filename of the output image, and it is optional!! 
    First value of return : total pixel of rusty part(red part)
    Second value of return : total pixel of towel part(white part)
    Thrid value of return : the percentage of rusty
    left image : The original image with enhancement
    middle image : The image include the part of tower(find the part of white)
    right image : The image include the part of rust(find the part of red and purple)
    '''
    if not os.path.isfile(input_img):
        print ("Fail to open ",end= ' ')
        print (input_img)
        return None,None,None
    
    img = cv2.imread(input_img, 1) #Read the image as img

    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  #Transform to RGB, the structure of PIT
    enh_col = ImageEnhance.Color(image)  #Color enhencement
    color = 3.0
    image_colored1 = enh_col.enhance(color) #Transform back to BGR, the structure of open CV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Create a HSV one    
    
    
    #Set different boundaries for different shades of rust
    boundaries1 = [([0, 0, 221],[180, 30, 255])]#white
    boundaries2 = [([0, 43, 46],[10, 255, 255])]#red 1
    boundaries5 = [([125, 43, 46],[180, 255, 255])]#red 2
    pixal_white = 0
    #detec all range white 
    for (lower1, upper1) in boundaries1:
        lower1 = np.array(lower1, dtype = "uint8")
        upper1 = np.array(upper1, dtype = "uint8")
        mask = cv2.inRange(img_hsv, lower1, upper1) #Finding all of the points in range WHITE
        for k in mask:
            for y in k:
                if y is not 0:
                    pixal_white = pixal_white+1 #calculate the part of white
        white_part = cv2.copyTo(img_hsv,mask)
        white_part = cv2.cvtColor(white_part, cv2.COLOR_HSV2BGR)
    
    #detec all range red 1
    for (lower2, upper2) in boundaries2:
        lower2 = np.array(lower2, dtype = "uint8")
        upper2 = np.array(upper2, dtype = "uint8")
        mask = cv2.inRange(img_hsv, lower2, upper2) #Finding all of the points in range RED
        pixal_red_1 = single_np(mask,255)
        red1_part = cv2.bitwise_and(img,img_hsv, mask = mask)
        red1_part = cv2.cvtColor(red1_part, cv2.COLOR_HSV2BGR)

    #detec all range red 2
    for (lower3, upper3) in boundaries2:
        lower3 = np.array(lower3, dtype = "uint8")
        upper3 = np.array(upper3, dtype = "uint8")
        mask = cv2.inRange(img_hsv, lower3, upper3) #Finding all of the points in range RED
        pixal_red_2 = single_np(mask,255)
        red2_part = cv2.bitwise_and(img,img_hsv, mask = mask)
        red2_part = cv2.cvtColor(red2_part, cv2.COLOR_HSV2BGR)
    
    red_part = red1_part + red2_part #The rusty part
    pixal_red = pixal_red_1 + pixal_red_2 #calculate the total pixal of rust part
    
    result1 = cv2.addWeighted(img, 0.1, red_part, 0.9, 0)  #融合圖片
    result = np.concatenate([img, white_part , result1], axis=1) #拚接圖片
    percentage = pixal_red / (pixal_white + pixal_red)
    #將拼接好的檔案丟到分類好的路徑資料夾下, optional
    if (output_img != 'no need') :
        try:
            cv2.imwrite(output_img,result)
        except:
            print ('Fail to output image to', end =' ')
            print (output_img) 
    return pixal_red,pixal_white,percentage



# x,y,z = rust_identification ('D:/Users\eric\Desktop\data/image/45.png','D:/Users\eric\Desktop/45.png')
# print (x) 
# print (y)
# print (z)


