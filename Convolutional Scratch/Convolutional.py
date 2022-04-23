import cv2
import numpy as np

# covulation process rotates the image matrix by 180 degree
def conv_transform(image):
    image_copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] =image [image.shape[0]-i-1] [image.shape [1]-j-1]
    return image_copy
    
kernel = np.ones((3, 3), np.float32)/9

# print(kernel) # 3 x 3

kernel_h = kernel.shape[0] # 3
kernel_w = kernel.shape[1] # 3

h = kernel_h//2 # 1
w = kernel_w//2 # 1

# print(kernel)
# print()
# print(conv_transform(kernel))

def conv(image, kernel):
    # The image will be grayscale, otherwise there will be confusion wi
    kernel = conv_transform(kernel) # Rotate 
    image_h = image.shape [0]   #7
    image_w = image.shape [1]   #7

    kernel_h = kernel.shape [0] #3
    kernel_w = kernel.shape [1] #3
    
    h = kernel_h//2
    w = kernel_w//2
    
    image_conv = np.zeros(image.shape)
    
    for i in range(h, image_h-h):
        for j in range(w, image_w-w):
            sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel [m] [n]*image[i-h-m] [j-w-n]
                    
            image_conv[i] [j] = sum
               
    return image_conv

img = cv2.imread('image.png',0) # Gray Scaled Image

output = conv(img,kernel)

cv2. imshow('Convolved image', output)
