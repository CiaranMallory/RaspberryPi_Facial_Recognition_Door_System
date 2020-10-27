import numpy as np
import face_recognition as fr
import cv2
import RPi.GPIO as GPIO
from time import sleep

video_capture = cv2.VideoCapture(0)

ciaran_image = fr.load_image_file("Ciarans_Face_test.jpg")
ciaran_face_encoding = fr.face_encodings(ciaran_image)[0]

known_face_encodings = [ciaran_face_encoding]
known_face_names = ["Ciaran"]

while True:
	ret, frame = video_capture.read()

	# Converts frame color to rgb so OpenCV can recognize it
	rgb_frame = frame [:, :, ::-1]

	face_locations = fr.face_locations(rgb_frame)
	face_encodings = fr.face_encodings(rgb_frame, face_locations)

	for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
		matches = fr.compare_faces(known_face_encodings, face_encodings)

		name = "Unknown"

		face_distances = fr.face_distance(known_face_encodings, face_encodings)

		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]

		cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255,255,255), 1)

	cv2.imgshow('Face Detection', frame)

	if (name == known_face_names): # Checking if known face is detected
		# Initializing servo motor to pin 3
		servoPin = 03
		# Setting naming mode for Raspberry Pi
		GPIO.setmode(GPIO.BOARD)
		# Setting pin 3 as an output for servo motor
		GPIO.setup(servoPin, GPIO.OUT)

		PWM = GPIO.PWM(servoPin, 50) # Set pin PWM at 50Hz
		PWM.start(0) # Set duty cycle to 0 to prevent motor moving on startup

		def SetAngle(angle):
			# Function for setting angle for the motor to go to
			duty = angle / 18 + 2
			GPIO.output(servoPin, True)
			PWM.ChangeDutyCycle(duty)
			sleep(1) # Delay 1 second
			GPIO.output(servoPin, False)
			PWM.ChangeDutyCycle(0)

		SetAngle(90) # Open door
		sleep(10) # Wait 10 seconds
		SetAngle(0) # Close door


video_capture.release()
cv2.destroyAllWindows()
PWM.stop()
GPIO.cleanup()