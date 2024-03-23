import cv2
import numpy
import utils

#prepoaring the image
img = cv2.imread('resources/1.webp') #reading the image with the cv2 inbuilt function
img = cv2.resize(img, (225, 225)) # resizing the image to 450X450
imgBlank = numpy.zeros((225, 225, 3), numpy.uint8) #creating a blank image
imgTreshold = utils.preProcess(img) #function in the utils file

#finding all the contours
imgContours = img.copy() #copying the img 
imgBigContours = img.copy() #copying the img
contours, hierarchy = cv2.findContours(imgTreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) #drawing detected contours

biggest, maxArea = utils.biggestContour(contours) # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest = utils.reorder(biggest)
    cv2.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = numpy.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = numpy.float32([[0, 0],[225, 0], [0, 225],[225, 225]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (225, 225))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

imageArray = ([img, imgTreshold, imgContours, imgBigContours], [imgWarpColored, imgBlank, imgBlank, imgBlank])
stackedImage = utils.stackImages(imageArray, 1)
cv2.imshow('output', stackedImage)

cv2.waitKey(0)
