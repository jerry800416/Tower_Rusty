import cv2                      
import os 
import numpy as np
import matplotlib.pyplot as plt


output_dir = './output_img/'	        
if not os.path.exists(output_dir):
	os.mkdir(output_dir)	

img_file = './image/'
filename = '112.PNG'

img_bgr = cv2.imread(img_file+filename,cv2.IMREAD_COLOR)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
# (L,A,B) = cv2.split(img_lab)
# print(L)
# print(L.shape)
# L = np.int64(L>200)
# print(L.shape)
# AB_test = cv2.merge([L,A,B])
print(img_hsv)

# 分割lab
img_l = img_lab[..., 0]
img_a = img_lab[..., 1]
img_b = img_lab[..., 2]
img_test = img_lab[...,1]+img_lab[...,2]+img_lab[..., 0]
# print(img_a.shape)
# print(img_c.shape)


#change img(2D) to 1D
img_lab_1D = img_a.reshape((img_a.shape[0]*img_a.shape[1],1))
img_lab_1D = np.float32(img_lab_1D)

img_lab_2D = img_a.reshape((img_b.shape[0]*img_b.shape[1],1))
img_lab_2D = np.float32(img_lab_2D)


# print(img_lab_1D.shape)
#define criteria = (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#set flags: hou to choose the initial center
#---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

# apply kmenas
compactness,labels,centers = cv2.kmeans(img_lab_1D,4,None,criteria,10,flags)
compactness2,labels2,centers2 = cv2.kmeans(img_lab_2D,4,None,criteria,10,flags)



img2 = labels.reshape((img_a.shape[0],img_lab.shape[1]))
img3 = labels2.reshape((img_b.shape[0],img_lab.shape[1]))

# print(img2)
# print(img2.shape)

plt.subplot(221)
plt.imshow(img_test,'gray')
plt.title('test')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(img_lab,'gray')
plt.title('LAB')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(img2,'gray')
plt.title('Akmeans')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(img3,'gray')
plt.title('Bkmeans')
plt.xticks([])
plt.yticks([])


plt.show()


# cv2.imshow('BGR',img_bgr)
# cv2.imshow('LAB',img_lab)
# cv2.imshow('L',img_l)
# cv2.imshow('A',img_a)
# cv2.imshow('B',img_b)
# cv2.waitKey()
# cv2.destroyAllWindows()
