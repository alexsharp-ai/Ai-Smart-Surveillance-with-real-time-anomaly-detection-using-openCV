import cv2
from spot_diff import spot_diff
import time
import numpy as np
from datetime import datetime



# from pushbullet import Pushbullet
# import smtplib

# sender_email = "thusharamohan18@gmail.com"
# receiver_email = "thusharamohan7.18@gmail.com"

# subject = "Alert"
# message = "Object theft detected."

# text = f"Subject: {subject}\n\n{message}"
# server = smtplib.SMTP("smtp.gmail.com", 587)
# server.starttls()
# server.login(sender_email, "xiwhpbdukgdasamj")

# # Replace with your Pushbullet API key
# API_KEY = "o.JT2dYLr7Il5WFkCJ1BAY85VNK0GoRDWO"

# # Create a Pushbullet instance
# pb = Pushbullet(API_KEY)

# def send_push_notification(title, message):
#     # Send a push notification
#     push = pb.push_note(title, message)
#     print("Notification sent!")

def find_motion():

	motion_detected = False
	is_start_done = False

	# Use RTSP stream
	cap = cv2.VideoCapture("rtsp://B48mfZuY:1ilmri3ObV99IADP@192.168.2.12:554/live/ch0")
	
	# Check if camera opened successfully
	if not cap.isOpened():
	    print("Error: Could not open RTSP stream.")
	    return

	check = []
	
	print("waiting for 2 seconds")
	time.sleep(2)
	frame1 = cap.read()

	_, frm1 = cap.read()
	frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

	
	while True:
		_, frm2 = cap.read()
		frm2 = cv2.cvtColor(frm2, cv2.COLOR_BGR2GRAY)

		diff = cv2.absdiff(frm1, frm2)

		_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

		contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

		#look at it
		contors = [c for c in contors if cv2.contourArea(c) > 25]


		if len(contors) > 5:
			cv2.putText(frm2, "motion detected", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			cv2.putText(frm2, f"{datetime.now().strftime('%y-%m-%d-%H:%M:%S')}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
			motion_detected = True
			
			is_start_done = False

		elif motion_detected and len(contors) < 3:
			if (is_start_done) == False:
				start = time.time()
				is_start_done = True
				end = time.time()

			end = time.time()

			print(end-start)
			if (end - start) > 4:
				frame2 = cap.read()
				# cap.release()
				# cv2.destroyAllWindows()
				x = spot_diff(frame1, frame2)
				if x == 0:
					print("running again")
					return

				else:
					print("found motion")
					return

		# else:
		# 	cv2.putText(frm2, "no motion detected", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

		cv2.imshow("Motion Detection", frm2)
		

		_, frm1 = cap.read()
		frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

		if cv2.waitKey(1) == 27:  # ESC key to exit
			cap.release()
			cv2.destroyAllWindows()
			break

	return
