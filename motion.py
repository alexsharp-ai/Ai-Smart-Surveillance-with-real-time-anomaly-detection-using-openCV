import cv2
import numpy as np
import time
from datetime import datetime

def noise():
    cap = cv2.VideoCapture("rtsp://B48mfZuY:1ilmri3ObV99IADP@192.168.2.12:554/live/ch0")  # Use RTSP stream
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return
    
    # Wait for camera to initialize
    time.sleep(2)
    
    _, base_frame = cap.read()
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.GaussianBlur(base_gray, (21, 21), 0)
    
    print("Camera initialized. Monitoring for noise...")
    
    while True:
        _, current_frame = cap.read()
        if current_frame is None:
            break
            
        # Convert to grayscale and apply blur to reduce noise
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        # Calculate difference between base frame and current frame
        frame_diff = cv2.absdiff(base_gray, gray_frame)
        
        # Apply threshold
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate threshold image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Loop over the contours
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Ignore small contours
                continue
                
            # Compute the bounding box for the contour and draw it
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text showing noise detection
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(current_frame, "Noise Detected!", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(current_frame, timestamp, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("Noise Detection", current_frame)
        
        # Update base frame occasionally
        if time.time() % 10 < 0.1:  # Update roughly every 10 seconds
            _, base_frame = cap.read()
            base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
            base_gray = cv2.GaussianBlur(base_gray, (21, 21), 0)
        
        # If 'ESC' is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    return