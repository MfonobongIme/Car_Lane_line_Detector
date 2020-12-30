#import the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('highway.png') #read the image
img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)
height = img.shape[0]
width = img.shape[1]

#define our region of interest to crop out the parts of the image we dont need
region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]

#mask every other thing other than our region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2] #number of channels in the image
    match_mask_color = (255,) #* channel_count#create a match color with the same color channel count
    cv2.fillPoly(mask, vertices, match_mask_color)#fill inside the polygon(region of interest)
    masked_image = cv2.bitwise_and(img, mask) #return the image only where the mask pixel matches

    return masked_image

#draw lines on image function
def draw_the_lines(img, lines):
    img = np.copy(img) #copy image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8) #create a blank image that matches the original image size

    #loop around the lines vector and draw all the lines found
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness = 8)

    #merge the blank image with the lines into the original image
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convert image to grayscale
edges = cv2.Canny(gray, 100, 200)
cropped_image = region_of_interest(edges,
                                   np.array([region_of_interest_vertices], np.int32))

lines = cv2.HoughLinesP(cropped_image, rho = 6,
                        theta = np.pi / 60,
                        threshold = 160,
                        lines = np.array([]),
                        minLineLength=40,
                        maxLineGap=25) #use hpughlinesp method to get the lines from the edged image into the hough space

image_with_lines = draw_the_lines(img, lines)
plt.imshow(image_with_lines)
plt.show()








