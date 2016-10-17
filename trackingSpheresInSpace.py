# USAGE
# change code with correct color values for the spheres that are going to be used for tracking
# input room size and cameras location (two cameras needed)
# calbrate: place spheres at different distances and call calibration

# import the necessary packages
from collections import deque
from math import sqrt
import numpy as np
import argparse
import imutils
import cv2
import socket

#----------- IF COORDINATES ARE TO BE SENT SOMEWHERE VIA SOCKETS ------
# UDP_IP = "127.0.0.1"
# UDP_PORT = 5005
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

# ------------------ FUNCTIONS FOR RADIUS INTERSECTION ----------------------

def IntersectPoints(P0, P1, r0, r1):
    if type(P0) != complex or type(P1) != complex:
        raise TypeError("P0 and P1 must be complex types")
    # d = distance
    d = sqrt((P1.real - P0.real)**2 + (P1.imag - P0.imag)**2)
    # n**2 in Python means "n to the power of 2"
    # note: d = a + b

    if d > (r0 + r1):
        return False
    elif d < abs(r0 - r1):
        return True
    elif d == 0:
        return True
    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        b = d - a
        h = sqrt(r0**2 - a**2)
        P2 = P0 + a * (P1 - P0) / d

        i1x = P2.real + h * (P1.imag - P0.imag) / d
        i1y = P2.imag - h * (P1.real - P0.real) / d
        i2x = P2.real - h * (P1.imag - P0.imag) / d
        i2y = P2.imag + h * (P1.real - P0.real) / d

        i1 = complex(i1x, i1y)
        i2 = complex(i2x, i2y)

        #return [i1, i2]
        return [[i1x, i1y], [i2x, i2y]]

def CompToStr(c):
    return "(" + str(c.real) + ", " + str(c.imag) + ")"

def PairToStr(p):
    return CompToStr(p[0]) + " , " + CompToStr(p[1])


# ------------------ MAIN ---------------------------------------------------

# ask for initial information - REPLACED BY HARD CODED INPUT BELOW
camLeftX = int(raw_input('Type position of left camera on the X axis: '))
camLeftY = int(raw_input('Type position of left camera on the Y axis: '))
camRightX = int(raw_input('Type position of right camera on the X axis: '))
camRightY = int(raw_input('Type position of right camera on the Y axis: '))
roomWidth = int(raw_input('Type width of the room (were cameras are): '))
roomLength = int(raw_input('Type length of the room: '))


# define the lower and upper boundaries of the mask
# for balls of 4 colors in the HSV color space
# LEGEND: 0-GREEN, 1-RED, 2-BLUE, 3-MAGENTA
hsvLowerL = []
hsvUpperL = []
hsvLowerR = []
hsvUpperR = []

# CHANGE THESE FOR APPROPRIATE VALUES
#green
hsvLowerL.append((72, 89, 97))
hsvUpperL.append((96, 255, 255))
hsvLowerR.append((74, 146, 144))
hsvUpperR.append((94, 255, 255))
#red
hsvLowerL.append((0, 248, 54))
hsvUpperL.append((5, 255, 255))
hsvLowerR.append((0, 255, 116))
hsvUpperR.append((3, 255, 255))
#blue
hsvLowerL.append((101, 235, 213))
hsvUpperL.append((121, 255, 255))
hsvLowerR.append((96, 250, 205))
hsvUpperR.append((124, 255, 255))
#cyan
hsvLowerL.append((96, 156, 185))
hsvUpperL.append((107, 255, 255))
hsvLowerR.append((89, 0, 151))
hsvUpperR.append((119, 243, 255))

#rgb(bgr) colors for a max of 6 possible balls: green, red, blue, cyan, magenta, yellow
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# initialize calibration arrays
ballDistanceLeft = []
ballRadiusLeft = []
ballDistanceRight = []
ballRadiusRight = []
personCoordinatesArray = []
#initialize other arrays
maskLeft = []
maskRight = []
cntsRight = []
cntsLeft = []

for index in range(len(hsvLowerL)):
	ballDistanceLeft.append(-1000)
	ballRadiusLeft.append(0)
	ballDistanceRight.append(-1000)
	ballRadiusRight.append(0)
	personCoordinatesArray.append([0,0])
	maskLeft.append(None)
	maskRight.append(None)
	cntsRight.append(None)
	cntsLeft.append(None)

# calibration reference list - REPLACED BY HARD CODED INPUT BELOW
calibrationRefLeft = []
calibrationRefRight = []

# grab the reference to the webcam
# id cameras while being front to front with them
cameraLeft = cv2.VideoCapture(0)
cameraRight = cv2.VideoCapture(2)
#have cameras capturing at their best resolution
cameraLeft.set(3,1920)
cameraLeft.set(4,1080)
cameraRight.set(3,1920)
cameraRight.set(4,1080)
# set calibration parameters for camera logitech c920
camera_matrix_left = np.array([[2096, 0.0, 969.637858], [0.0, 1890, 559.744399], [0.0, 0.0, 1.0]])
dist_coefs_left = np.array([1, -0.22255446, 0.00056116, -0.00045448, 0.03984153])
# set calibration parameters for camera logitech c615
camera_matrix_right = np.array([[2692, 0.0, 927.457968], [0.0, 2815, 528.121566], [0.0, 0.0, 1.0]])
dist_coefs_right = np.array([1, 0.327782679, -0.00132016178, -0.000181788821, -0.423082747])

# keep looping
while True:
	# grab the current frame
	(grabbedLeft, frameLeft) = cameraLeft.read()
	(grabbedRight, frameRight) = cameraRight.read()

	#calibrate image from camera c920
	hl,  wl = frameLeft.shape[:2]
	newcameramtx_left, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_left, dist_coefs_left, (wl, hl), 1, (wl, hl))
	frameLeft = cv2.undistort(frameLeft, camera_matrix_left, dist_coefs_left, None, newcameramtx_left)
	#calibrate image from camera c615
	hr,  wr = frameRight.shape[:2]
	newcameramtx_right, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_right, dist_coefs_right, (wr, hr), 1, (wr, hr))
	frameRight = cv2.undistort(frameRight, camera_matrix_right, dist_coefs_right, None, newcameramtx_right)

	# convert frame to the HSV color space
	hsvLeft = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2HSV)
	hsvRight = cv2.cvtColor(frameRight, cv2.COLOR_BGR2HSV)

	#perform detection, tracking and registration for each ball
	for index in range(len(hsvLowerL)):
		# construct a mask for each color, then perform a series of dilations
		# and erosions to remove any small blobs left in the mask
		maskLeft[index] = cv2.inRange(hsvLeft, hsvLowerL[index], hsvUpperL[index])
		maskLeft[index] = cv2.erode(maskLeft[index], None, iterations=2)
		maskLeft[index] = cv2.dilate(maskLeft[index], None, iterations=2)

		maskRight[index] = cv2.inRange(hsvRight, hsvLowerR[index], hsvUpperR[index])
		maskRight[index] = cv2.erode(maskRight[index], None, iterations=2)
		maskRight[index] = cv2.dilate(maskRight[index], None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cntsRight[index] = cv2.findContours(maskRight[index].copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		cntsLeft[index] = cv2.findContours(maskLeft[index].copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		# >>> TRACKING AND FINDING DISTANCE FOR LEFT CAMERA----------------------------------
		# only proceed if at least one contour was found
		if len(cntsLeft[index]) == 0:
			ballDistanceLeft[index] = -1000
		if len(cntsLeft[index]) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cntsLeft[index], key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			# M = cv2.moments(c)
			# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 3:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frameLeft, (int(x), int(y)), int(radius),
					colors[index], 2)
				# cv2.circle(frameLeft, center, 5, (0, 0, 255), -1)

			# calculate ballDistance
			ballRadiusLeft[index] = radius
			if len(calibrationRefLeft) == 0:
				#this 'nothing' variable is here just to prevent the elif to run while it's not ready
				nothing=0
			elif ballRadiusLeft[index] <= calibrationRefLeft[0][0]:
				ballDistanceLeft[index] = calibrationRefLeft[0][1]
			elif ballRadiusLeft[index] >= calibrationRefLeft[len(calibrationRefLeft)-1][0]:
				ballDistanceLeft[index] = calibrationRefLeft[len(calibrationRefLeft)-1][1]
			else:
				for j in range(len(calibrationRefLeft)):
					if ballRadiusLeft[index] == calibrationRefLeft[j][0]:
						ballDistanceLeft[index] = calibrationRefLeft[j][1]
					elif j < len(calibrationRefLeft) -1:
						if (ballRadiusLeft[index] > calibrationRefLeft[j][0]) & (ballRadiusLeft[index] < calibrationRefLeft[j+1][0]):
							diffCalibrationRadius = calibrationRefLeft[j+1][0] - calibrationRefLeft[j][0]
							diffCalibrationDistance = calibrationRefLeft[j][1] - calibrationRefLeft[j+1][1]
							diffActualRadius = ballRadiusLeft[index] - calibrationRefLeft[j][0]
							ballDistanceLeft[index] = calibrationRefLeft[j][1] - (diffCalibrationDistance * diffActualRadius/diffCalibrationRadius)

		# >>> TRACKING AND FINDING DISTANCE FOR RIGHT CAMERA----------------------------------
		# only proceed if at least one contour was found
		if len(cntsRight[index]) == 0:
			ballDistanceRight[index] = -1000
		if len(cntsRight[index]) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cntsRight[index], key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			# M = cv2.moments(c)
			# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 3:
				# draw the circle and centroid on the frame
				cv2.circle(frameRight, (int(x), int(y)), int(radius),
					colors[index], 2)
				# cv2.circle(frameRight, center, 5, (0, 0, 255), -1)

			# calculate ballDistance
			ballRadiusRight[index] = radius
			if len(calibrationRefRight) == 0:
				#this 'nothing' variable is here just to prevent the elif to run while it's not ready
				nothing = 0
			elif ballRadiusRight[index] <= calibrationRefRight[0][0]:
				ballDistanceRight[index] = calibrationRefRight[0][1]
			elif ballRadiusRight[index] >= calibrationRefRight[len(calibrationRefRight)-1][0]:
				ballDistanceRight[index] = calibrationRefRight[len(calibrationRefRight)-1][1]
			else:
				for j in range(len(calibrationRefRight)):
					if ballRadiusRight[index] == calibrationRefRight[j][0]:
						ballDistanceRight[index] = calibrationRefRight[j][1]
					elif j < len(calibrationRefRight) -1:
						if (ballRadiusRight[index] > calibrationRefRight[j][0]) & (ballRadiusRight[index] < calibrationRefRight[j+1][0]):
							diffCalibrationRadius = calibrationRefRight[j+1][0] - calibrationRefRight[j][0]
							diffCalibrationDistance = calibrationRefRight[j][1] - calibrationRefRight[j+1][1]
							diffActualRadius = ballRadiusRight[index] - calibrationRefRight[j][0]
							ballDistanceRight[index] = calibrationRefRight[j][1] - (diffCalibrationDistance * diffActualRadius/diffCalibrationRadius)

		# >>> FIND FINAL COORDINATES ----------------------------------------
		if ballDistanceLeft[index] != -1000 and ballDistanceRight[index] != -1000:
			ip = IntersectPoints
			# ballDistanceRight and ballDistanceLeft are in inverted positions so that coordinates are in pixels logic
			intersectPoints = ip(complex(camLeftX,camLeftY), complex(camRightX, camRightY), ballDistanceRight[index], ballDistanceLeft[index])
			# choose the coordinates where x and y are positive numbers
			if isinstance(intersectPoints, list):
				if intersectPoints[0][0] >= 0 and intersectPoints[0][1] >= 0:
					personCoordinatesArray[index] = intersectPoints[0]
				elif intersectPoints[1][0] >= 0 and intersectPoints[1][1] >= 0:
					personCoordinatesArray[index] = intersectPoints[1]

		if ballDistanceLeft[index] == -1000 and ballDistanceRight[index] == -1000:
			personCoordinatesArray[index] = [0, 0]

		if ballDistanceLeft[index] != -1000 and ballDistanceRight[index] == -1000:
			roomWidth = camRightY - camLeftY
			personCoordinatesArray[index] = [ballDistanceLeft[index], roomWidth*3/4]

		if ballDistanceLeft[index] == -1000 and ballDistanceRight[index] != -1000:
			roomWidth = camRightY - camLeftY
			personCoordinatesArray[index] = [ballDistanceRight[index], roomWidth/4]


	# show the frame to our screen
	frameLeft = imutils.resize(frameLeft, width=500)
	frameRight = imutils.resize(frameRight, width=500)
	cv2.imshow("FrameLeft", frameLeft)
	cv2.imshow("FrameRight", frameRight)
	key = cv2.waitKey(1) & 0xFF

	roomMap = cv2.imread("blank.jpg")
	roomMap = cv2.resize(roomMap, (roomLength, roomWidth))
	for index in range(len(hsvLowerL)):
		cv2.circle(roomMap, (int(personCoordinatesArray[index][0]), int(personCoordinatesArray[index][1])), 3, colors[index], 2)
	cv2.imshow("roomMap", roomMap)
	cv2.waitKey(1)

	# send coordinates as a string via socket. Balls separated by * and xy pairs by -
	for index in range(len(hsvLowerL)):
		if index == 0:
			message = str(int(personCoordinatesArray[index][0])) + "-" + str(int(personCoordinatesArray[index][1]))
		else:
			message = message + "*" + str(int(personCoordinatesArray[index][0])) + "-" + str(int(personCoordinatesArray[index][1]))
	sock.sendto(message, (UDP_IP, UDP_PORT))

	# ------------------------ DEFINE COMMANDS ------------------------------------------------

	#append calibration pair to cameraLeft when user clicks 'e'
	if key == ord("e"):
		distInput = raw_input('How distant (cm) is the green ball from the webcam on the left? ')
		calibrationPair = (ballRadiusLeft[0], int(distInput))
		calibrationRefLeft.append(calibrationPair)
		calibrationRefLeft.sort(key=lambda tup: tup[0])
		print calibrationRefLeft
	#append calibration pair to cameraLeft when user clicks 'r'
	if key == ord("r"):
		distInput = raw_input('How distant (cm) is the green ball from the webcam on the right? ')
		calibrationPair = (ballRadiusRight[0], int(distInput))
		calibrationRefRight.append(calibrationPair)
		calibrationRefRight.sort(key=lambda tup: tup[0])
		print calibrationRefRight

	# if the 'd' key is pressed, the left calibration reference list is printed
	if key == ord("d"):
		print ("Radius is: ", ballRadiusLeft[0], "| Distance is:", ballDistanceLeft[0])
	# if the 'f' key is pressed, the right calibration reference list is printed
	if key == ord("f"):
		print ("Radius is: ", ballRadiusRight[0], "| Distance is:", ballDistanceRight[0])
	# if the 'a' key is pressed, the current coordinate is printed
	if key == ord("a"):
		print ("Current ball coordinates is:", personCoordinatesArray[0])
	if key == ord("w"):
		f = open( 'calibration.txt', 'w' )
		f.write( 'calibrationRefLeft is = ' + repr(calibrationRefLeft) + '\n' + 'calibrationRefRight is = ' + repr(calibrationRefRight) + '\n' )
		f.close()

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
cameraLeft.release()
cameraRight.release()
cv2.destroyAllWindows()