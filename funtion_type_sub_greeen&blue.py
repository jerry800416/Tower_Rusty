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

def detec_color(boundaries,img_hsv):
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(img_hsv, lower, upper) #Finding all of the points in range
        output = cv2.copyTo(img_hsv,mask = mask)
        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    pixal_part = single_np(mask,255)
    return output, pixal_part 

def rust_identification (input_img ,output_img = 'no need'):
    '''
    input_img : The filename of the input image 
    output_img : The filename of the output image, and it is optional!! 
    First value of return : total pixel of rusty part
    Second value of return : total pixel of towel part
    Thrid value of return : the percentage of rusty
    left image : The original image with enhancement
    middle image : The image include the part of tower(eliminate the part of green and blue)
    right image : The image include the part of rust(find the part of red and purple)
    '''
    if not os.path.isfile(input_img):
        print ("Fail to open ",end= ' ')
        print (input_img)
        return None,None,None
    
    img = cv2.imread(input_img, 1) #Read the image as img

    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #Transform to RGB, the structure of PIT
    enh_col = ImageEnhance.Color(image) #Color enhencement
    color = 2.0
    image_colored1 = enh_col.enhance(color) 
    img = cv2.cvtColor(np.asarray(image_colored1),cv2.COLOR_RGB2BGR) #Transform back to BGR, the structure of open CV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Create a HSV one       
    
    #Set different boundaries for different shades of rust
    boundaries1 = [([35, 43, 46],[77, 255, 255])]#green    
    boundaries2 = [([0, 43, 46],[10, 255, 255])]#red 1
    boundaries3 = [([100, 43, 46],[124, 255, 255])]#blue
    boundaries4 = [([78, 43, 46],[99, 255, 255])]#cyan-blue
    boundaries5 = [([125, 43, 46],[180, 255, 255])]#red 2
    pixal_tower = 0 #Calculating total pixal of tower
    #Detect each color
    green_part,dummy = detec_color(boundaries1,img_hsv) #detec all range green     
    blue_part,dummy = detec_color(boundaries3,img_hsv) #detec all range blue
    cyan_blue_part,dummy = detec_color(boundaries4,img_hsv) #detec all range cyan-blue
    red1_part,pixal_red_1 = detec_color(boundaries2,img_hsv) #detec all range red 1    
    red2_part,pixal_red_2 = detec_color(boundaries5,img_hsv) #detec all range red 2

    red_part = red1_part + red2_part #The rusty part
    pixal_red = pixal_red_1 + pixal_red_2 #calculate the total pixal of rust part

    temp = cv2.subtract(img,green_part) #Subtract the green part
    temp = cv2.subtract(temp,blue_part) #Subtract the blue part
    result1 = cv2.subtract(temp,cyan_blue_part) #Subtract the Syan-blue part       

    dst1 = cv2.bilateralFilter(result1, 30,200,200)#模糊
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) 
    dst = cv2.filter2D(dst1, -1, kernel=kernel)#銳化

    imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) #灰階
    ret, thresh = cv2.threshold(imgray, 127, 255, 0) #Determine the threshold
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Set the initial condition of contours detecting
    
    for cnt in contours:
        area = cv2.contourArea(cnt) #Finding the area in the contour
        cv2.drawContours(dst,[cnt],0,(0,255,0),1) #Draw the contour 
        pixal_tower+=area #The total pixal of tower

    percentage = pixal_red/pixal_tower
    result2 = cv2.addWeighted(img, 0.1, red_part, 0.9, 0)  #融合圖片
    result = np.concatenate([img, dst, result2], axis=1) #拚接

    #將拼接好的檔案丟到分類好的路徑資料夾下, optional
    if (output_img != 'no need') :
        try:
            cv2.imwrite(output_img,result)
        except:
            print ('Fail to output image to', end =' ')
            print (output_img) 
    return pixal_red,pixal_tower,percentage
    




