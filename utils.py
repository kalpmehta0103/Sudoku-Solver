import cv2;
import numpy

#preprocessing the image
def preProcess(img) : 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the image into gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) #blurring the image
    imgTreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2) #Applying adaptive treshold
    return imgTreshold

# stacking the image
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = numpy.zeros((height, width, 3), numpy.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = numpy.hstack(imgArray[x])
            hor_con[x] = numpy.concatenate(imgArray[x])
        ver = numpy.vstack(hor)
        ver_con = numpy.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= numpy.hstack(imgArray)
        hor_con= numpy.concatenate(imgArray)
        ver = hor
    return ver

# finding the biggest contour
def biggestContour(contours):
    biggest = numpy.array([])
    max_area = 0 #defining maxArea as 0
    for i in contours: #looping through all the contours
        area = cv2.contourArea(i) #finding area of contours
        if area > 50: #making sure contours have area greater than certain treshold
            peri = cv2.arcLength(i, True) #finding the perimeter of contours
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) #reducing the number of contours point 
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

#reordering the contours point
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = numpy.zeros((4, 1, 2), dtype=numpy.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[numpy.argmin(add)]
    myPointsNew[3] =myPoints[numpy.argmax(add)]
    diff = numpy.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[numpy.argmin(diff)]
    myPointsNew[2] = myPoints[numpy.argmax(diff)]
    return myPointsNew