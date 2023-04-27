#This python script is to detect aruco markers and then calculate its pose and highlight it

import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import time

#creating a dictionary of aruco markers with different dimensions and tag size
ARUCO_DICT = {
	"DICT_4X4_50":aruco.DICT_4X4_50,
	"DICT_4X4_100":aruco.DICT_4X4_100,
	"DICT_4X4_250":aruco.DICT_4X4_250,
	"DICT_4X4_1000":aruco.DICT_4X4_1000,
	"DICT_5X5_50":aruco.DICT_5X5_50,
	"DICT_5X5_100":aruco.DICT_5X5_100,
	"DICT_5X5_250":aruco.DICT_5X5_250,
	"DICT_5X5_1000":aruco.DICT_5X5_1000,
	"DICT_6X6_50":aruco.DICT_6X6_50,
	"DICT_6X6_100":aruco.DICT_6X6_100,
	"DICT_6X6_250":aruco.DICT_6X6_250,
	"DICT_6X6_1000":aruco.DICT_6X6_1000,
	"DICT_7X7_50":aruco.DICT_7X7_50,
	"DICT_7X7_100":aruco.DICT_7X7_100,
	"DICT_7X7_250":aruco.DICT_7X7_250,
	"DICT_7X7_1000":aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL":aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5":aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9":aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10":aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11":aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
	
	if len(corners) > 0:
				#if corners exist
		ids = ids.flatten() #make the id matrix 1d
		
		for (markerCorner, markerID) in zip(corners, ids):
						
			corners = markerCorner.reshape((4, 2))      #converting each corner coordinates into a 4X2 matrix
			(topLeft, topRight, bottomRight, bottomLeft) = corners     #start from top left clockwise
						

			#coverting into int as opencv only works with integers

			topRight = (int(topRight[0]), int(topRight[1])) 
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))
						
			#drawing rectangles
			cv2.line(image, topLeft, topRight, (0, 255, 0), 2) 
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			#midpoints of the object and drawing a circle

			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image



def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
	

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting the frame to grayscale
	aruco_dict =aruco.Dictionary_get(aruco_dict_type)
	parameters =aruco.DetectorParameters_create()


	corners, ids, rejected_img_points =aruco.detectMarkers(gray,aruco_dict,parameters=parameters,cameraMatrix=matrix_coefficients,distCoeff=distortion_coefficients)

		
	if len(corners) > 0: #if corners exist
		for i in range(0, len(ids)):
		   
		   #This function returns the rotaional and translational vectors(POSE) of detected aruco markers

			rvec, tvec, markerPoints =aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
																	   distortion_coefficients)
			
			#drawing out the markers and axis on the aruco markers
		aruco.drawDetectedMarkers(frame, corners) 
		aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

	return frame


	
#getting the required aruco marker dimensions and tag size
user_input =input("Enter the aruco marker size axb and tag size : ").split()
aruco_type=f"DICT_{user_input[0]}X{user_input[1]}_{user_input[2]}"

arucoDict =aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams =aruco.DetectorParameters_create()

#these two values are obtained after caliberating the camera with sample images like chessboard

intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
distortion = np.array((-0.43948,0.18514,0,0))

#opens live videocam
cap = cv2.VideoCapture(0)

#setiing to image to hd resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while cap.isOpened():
	#returning frames
	ret, img = cap.read()
	
	if not ret:
		break
	
	h,w,_=img.shape
	
	#step 1: Detecting the required aruco markers from the image 
	
	
	width=1000
	height=int(width*(h/w))
	
	#resizing the image
	resized_img=cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
	
	#detects the given aruco marker from the resized image

	corners,ids,rejected=aruco.detectMarkers(resized_img,arucoDict,parameters=arucoParams)
	
	#detected_markers=aruco_display(corners,ids,rejected,resized_img)
	
	#displaying the detected markers

	#cv2.imshow("Image",detected_markers)
	


	#Step 2: Calculating the pose 
	
	output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

	cv2.imshow('Estimated Pose', output)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'): #if q key is pressed the while loop will be terminated
		break

cap.release()
cv2.destroyAllWindows()
