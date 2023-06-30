import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
from playsound import playsound


dirname = os.path.dirname(__file__)
tapsound_path = os.path.join(dirname, 'TapBeep.wav')
alertsound_path = os.path.join(dirname, 'AlertBeep.wav')
bicep_ins_path = os.path.join(dirname, 'BicepInstruction.jpg')
plank_ins_path = os.path.join(dirname, 'PlankInstruction.jpg')
		
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

global mouseX,mouseY

mouseX = 0
mouseY = 0

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def calculateangle(a,b,c):
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	
	ba = a - b
	bc = c - b
	
	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	radians = np.arccos(cosine_angle)
	angle = np.abs(radians*180.0/np.pi)
	
	return angle

def tap(event,x,y,flags,param):
	global mouseX,mouseY
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y
		
def BicepModel():
	Lcounter = 0
	Rcounter = 0
	cap = cv2.VideoCapture(0)
	global mouseX,mouseY
	mouseX = 0
	mouseY = 0
	while (mouseX == 0 and mouseY == 0):
		# Generate blank image canvas in phone dimensions
		img = cv2.imread(bicep_ins_path)
		img = cv2.resize(img, (1200,2000))
		cv2.imshow('Bicep Instructions', img)
		cv2.setMouseCallback('Bicep Instructions',tap)
		cv2.waitKey(1)

	timer = 30


	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
		while cap.isOpened():
			ret, frame = cap.read()
			
			# Recolor image to RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			image = cv2.flip(image,1)
		  
			# Make detection
			results = pose.process(image)
		
			# Recolor back to BGR
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			
			# Extract landmarks
			try:
				landmarks = results.pose_landmarks.landmark
				
				# Get coordinates
				Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
				Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
				Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
				
				Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
				Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
				Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
				
				# Calculate angle
				Langle = calculateangle(Lshoulder, Lelbow, Lwrist)
				Rangle = calculateangle(Rshoulder, Relbow, Rwrist)
				
				# Visualize angle
				#cv2.putText(image, str(round(Langle)), 
				#			   tuple(np.multiply(Lshoulder, [1280, 480]).astype(int)), 
				#			   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA
				#					)
				#cv2.putText(image, "ANGLE", 
				#			   tuple(np.multiply(Lshoulder, [1280, 480]).astype(int)+(0,40)), 
				#			   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
				#					)
				
				#cv2.putText(image, str(round(Rangle)), 
				#			   tuple(np.multiply(Rshoulder, [1280, 480]).astype(int)-(200,0)), 
				#			   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA
				#					)
				#cv2.putText(image, "ANGLE", 
				#			   tuple(np.multiply(Rshoulder, [1280, 480]).astype(int)-(200,-40)), 
				#			   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
				#					)
				
				# Rep Count
				if Langle > 160:
					Lstage = "DOWN"
				if Langle < 40 and Lstage == "DOWN":
					Lstage = "UP"
					if(timer == -1):
						Lcounter += 1
					
				if Rangle > 160:
					Rstage = "DOWN"
				if Rangle < 40 and Rstage == "DOWN":
					Rstage = "UP"
					if(timer == -1):
						Rcounter += 1
					
			except:
				pass
			
			
			# Render detections
			mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
									mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
									mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
									 )
			# Generate blank image canvas in phone dimensions
			img = np.zeros((1000,600,3), np.uint8)
			# Paste webcam view ontop of canvas
			img[0:1000,0:600,:] = image[0:1000,600:1200,:]
			# Apply graphics
			cv2.rectangle(img,(0,0),(600,50),(255,255,255),-1)
			cv2.rectangle(img,(0,0),(10,1000),(255,255,255),-1)
			cv2.rectangle(img,(600,0),(590,1000),(255,255,255),-1)
			cv2.rectangle(img,(0,1000),(600,800),(255,255,255),-1)
			#Apply text
			# Left Rep data
			cv2.putText(img, 'LEFT REPS', (5,850), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(img, str(Rcounter),(100,920),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
			
			# Right Rep data
			cv2.putText(img, 'RIGHT REPS', (310,850), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(img, str(Lcounter),(420,920),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

			# Back Button
			cv2.putText(img, '< BACK', (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,0), 1, cv2.LINE_AA)

			# Rest Button
			cv2.putText(img, 'RESET COUNT', (130,980), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			# Timer
			if(timer >= 0):
				cv2.putText(img, str(timer), (160,470), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,255,255), 3, cv2.LINE_AA)
				timer -= 1
			
			img = cv2.resize(img, (1200,2000))
			cv2.imshow('Mediapipe Feed', img)
			
			# Check mouse click position
			cv2.setMouseCallback('Mediapipe Feed',tap)

			if (mouseX > 276 and mouseX < 923):
				if(mouseY > 1883 and mouseY < 1963):
					mouseX = 0
					mouseY = 0
					Lcounter = 0
					Rcounter = 0
					timer = 30
			if (mouseX > 1 and mouseX < 100):
				if (mouseY > 1 and mouseY < 100):
					mouseX = 0
					mouseY = 0
					cap.release()
					#cv2.destroyAllWindows()
			if(mouseX != 0):
				print(mouseX)
				mouseX = 0
			if(mouseY != 0):
				print(mouseY)
				mouseY = 0
				
			#434,1872
			#688,1966

			if cv2.waitKey(10) & 0xFF == ord('r'):
				Rcounter = 0
				Lcounter = 0

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

def PlankModel():
	armstatus = "None"
	backstatus = "None"
	i=0
	cap = cv2.VideoCapture(0)
	global mouseX,mouseY
	mouseX = 0
	mouseY = 0
	while (mouseX == 0 and mouseY == 0):
		# Generate blank image canvas in phone dimensions
		img = cv2.imread(plank_ins_path)
		img = cv2.resize(img, (1200,2000))
		cv2.imshow('Plank Instructions', img)
		cv2.setMouseCallback('Plank Instructions',tap)
		cv2.waitKey(1)
		## Setup mediapipe instance
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
		while cap.isOpened():
			ret, frame = cap.read()
			
			# Recolor image to RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			image = cv2.flip(image,1)
		  
			# Make detection
			results = pose.process(image)
		
			# Recolor back to BGR
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
			# Extract landmarks
			try:
					landmarks = results.pose_landmarks.landmark
				
					# Get coordinates
					Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
					Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
					Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
					Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
					Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
					
					# Calculate angle
					backangle = calculateangle(Lshoulder, Lhip, Lknee)
					armangle = calculateangle(Lwrist, Lelbow, Lshoulder)
					
					# Visualize angle
					#cv2.putText(image, str(round(backangle)), 
					#			   tuple(np.multiply(Lshoulder, [1280, 480]).astype(int)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					#cv2.putText(image, "BACK ANGLE", 
					#			   tuple(np.multiply(Lshoulder, [1280, 480]).astype(int)+(0,40)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					
					#cv2.putText(image, str(round(armangle)), 
					#			   tuple(np.multiply(Lelbow, [1280, 480]).astype(int)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					#cv2.putText(image, "ARM ANGLE", 
					#			   tuple(np.multiply(Lelbow, [1280, 480]).astype(int)+(0,40)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					
					# Status
					if backangle > 160:
						backstatus = "HOLD"
					else:
						backstatus = "STRAIGHTEN"
						if(i<=30):
							playsound(alertsound_path)
							i = 100
						if (i>=30):
							i -= 1
					if armangle > 110:
						armstatus = "ARMS IN"
					elif armangle < 70:
						armstatus = "ARMS OUT"
					else:
						armstatus = "HOLD"
					
			except:
				pass
			
			
			# Render detections
			mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
									mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
									mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
									 )
			# Generate blank image canvas in phone dimensions
			img = np.zeros((1000,600,3), np.uint8)
			# Paste webcam view ontop of canvas
			img[0:1000,0:600,:] = image[0:1000,600:1200,:]
			# Apply graphics
			cv2.rectangle(img,(0,0),(600,50),(255,255,255),-1)
			cv2.rectangle(img,(0,0),(10,1000),(255,255,255),-1)
			cv2.rectangle(img,(600,0),(590,1000),(255,255,255),-1)
			cv2.rectangle(img,(0,1000),(600,800),(255,255,255),-1)
			#Apply text
			# Left Rep data
			cv2.putText(img, 'ARM STATE', (5,850), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(img, str(armstatus),(5,920),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
			
			# Right Rep data
			cv2.putText(img, 'BACK STATE', (310,850), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(img, str(backstatus),(320,920),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
			# Back Button
			cv2.putText(img, '< BACK', (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,0), 1, cv2.LINE_AA)

			img = cv2.resize(img, (1200,2000))
			cv2.imshow('Mediapipe Feed', img)
			
			# Check mouse click position
			cv2.setMouseCallback('Mediapipe Feed',tap)

			if (mouseX > 1 and mouseX < 100):
				if (mouseY > 1 and mouseY < 100):
					mouseX = 0
					mouseY = 0
					cap.release()
					#cv2.destroyAllWindows()

			if(mouseX != 0):
				print(mouseX)
				mouseX = 0
			if(mouseY != 0):
				print(mouseY)
				mouseY = 0
			
			if cv2.waitKey(10) & 0xFF == ord('r'):
				Rcounter = 0
				Lcounter = 0

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

def APTModel():
	hipstatus = "None"
	quadstatus = "None"
	cap = cv2.VideoCapture(0)
	global mouseX,mouseY
		## Setup mediapipe instance
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
		while cap.isOpened():
			ret, frame = cap.read()
			
			# Recolor image to RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			image = cv2.flip(image,1)
		  
			# Make detection
			results = pose.process(image)
		
			# Recolor back to BGR
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
			# Extract landmarks
			try:
					landmarks = results.pose_landmarks.landmark
				
					# Get coordinates
					Lshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
					Lhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
					Lknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
					Lankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
								
					# Calculate angle
					hipangle = calculateangle(Lshoulder, Lhip, Lknee)
					quadangle = calculateangle(Lhip, Lknee, Lankle)
					
					# Visualize angle
					#cv2.putText(image, str(round(hipangle)), 
					#			   tuple(np.multiply(Lhip, [1280, 480]).astype(int)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					#cv2.putText(image, "HIP ANGLE", 
					#			   tuple(np.multiply(Lhip, [1280, 480]).astype(int)+(0,40)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					
					#cv2.putText(image, str(round(quadangle)), 
					#			   tuple(np.multiply(Lankle, [1280, 480]).astype(int)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					#cv2.putText(image, "QUAD ANGLE", 
					#			   tuple(np.multiply(Lankle, [1280, 480]).astype(int)+(0,40)), 
					#			   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
					#					)
					
			
					# Status
					if hipangle > 160:
						hipstatus = "FINE"
					else:
						hipstatus = "TIGHT"
						
					if quadangle > 95:
						quadstatus = "TIGHT"
					else:
						quadstatus = "FINE"
						
							
			except:
				pass
			
			
			# Render detections
			mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
									mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
									mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
									 )
			# Generate blank image canvas in phone dimensions
			img = np.zeros((1000,600,3), np.uint8)
			# Paste webcam view ontop of canvas
			img[0:1000,0:600,:] = image[0:1000,600:1200,:]
			# Apply graphics
			cv2.rectangle(img,(0,0),(600,50),(255,255,255),-1)
			cv2.rectangle(img,(0,0),(10,1000),(255,255,255),-1)
			cv2.rectangle(img,(600,0),(590,1000),(255,255,255),-1)
			cv2.rectangle(img,(0,1000),(600,800),(255,255,255),-1)
			#Apply text
			# Left Rep data
			cv2.putText(img, 'HIP STATE', (5,850), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(img, str(hipstatus),(5,920),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
			
			# Right Rep data
			cv2.putText(img, 'QUAD STATE', (310,850), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(img, str(quadstatus),(320,920),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
			# Back Button
			cv2.putText(img, '< BACK', (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,0), 1, cv2.LINE_AA)
			
			img = cv2.resize(img, (1200,2000))
			cv2.imshow('Mediapipe Feed', img)
			
			# Check mouse click position
			cv2.setMouseCallback('Mediapipe Feed',tap)

			if (mouseX > 1 and mouseX < 100):
				if (mouseY > 1 and mouseY < 100):
					mouseX = 0
					mouseY = 0
					cap.release()
					#cv2.destroyAllWindows()

			if(mouseX != 0):
				print(mouseX)
				mouseX = 0
			if(mouseY != 0):
				print(mouseY)
				mouseY = 0
			
			if cv2.waitKey(10) & 0xFF == ord('r'):
				Rcounter = 0
				Lcounter = 0

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

def menu_main():
	print("enter main menu")
	global mouseX,mouseY
	# Generate blank image canvas in phone dimensions
	while True:
		#img = np.zeros((1000,600,3), np.uint8)
		#cv2.rectangle(img,(0,0),(600,1000),(255,255,255),-1)
		img = np.zeros((2000,1200,3), np.uint8)
		cv2.rectangle(img,(0,0),(1200,2000),(255,255,255),-1)
		# Apply graphics
		cv2.putText(img, 'Bicep Curl', (475,600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
		cv2.putText(img, 'Planking', (500,800), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
		cv2.putText(img, 'APT Diagnosis', (450,1000), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
		cv2.putText(img, 'QUIT', (550,1400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
		#img = cv2.resize(img, (1200,2000))
		cv2.imshow('Menu', img)
		cv2.setMouseCallback('Menu',tap)
		if (mouseX > 447 and mouseX < 734 and mouseY > 538 and mouseY < 615):
			print("B")
			mouseX = 0
			mouseY = 0
			BicepModel()
		if (mouseX > 471 and mouseX < 715 and mouseY > 756 and mouseY < 816):
			print("P")
			mouseX = 0
			mouseY = 0
			PlankModel()
		if (mouseX > 423 and mouseX < 779 and mouseY > 946 and mouseY < 1005):
			print("APT")
			mouseX = 0
			mouseY = 0
			APTModel()
		if (mouseX > 529 and mouseX < 665 and mouseY > 1353 and mouseY < 1406):
			print("QUIT")
			mouseX = 0
			mouseY = 0
			exit()
		mouseX = 0
		mouseY = 0

		cv2.waitKey(1)

def main():
	print("Program start")
	while True:
		print("Enter main")
		playsound(tapsound_path)
		menu_main()

if __name__ == '__main__':
	main()