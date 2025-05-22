import cv2
import numpy as np
import time
from datetime import datetime

def rect_noise():
    cap = cv2.VideoCapture(0)  # Use default webcam
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Wait for camera to initialize
    time.sleep(2)
    
    # Get first frame to determine dimensions
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture initial frame")
        return
        
    height, width = frame.shape[:2]
    
    # Define a rectangle area to monitor (center portion of frame)
    rect_top = int(height * 0.2)
    rect_bottom = int(height * 0.8)
    rect_left = int(width * 0.2)
    rect_right = int(width * 0.8)
    
    # Initial background model
    _, background = cap.read()
    background_roi = background[rect_top:rect_bottom, rect_left:rect_right]
    background_gray = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    
    print("Monitoring rectangle area for motion...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw rectangle on frame
        cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (0, 255, 0), 2)
        
        # Extract region of interest
        roi = frame[rect_top:rect_bottom, rect_left:rect_right]
        
        # Process ROI for motion detection
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        
        # Calculate difference between background and current ROI
        frame_diff = cv2.absdiff(background_gray, gray_roi)
        
        # Apply threshold
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate threshold image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        
        # Loop over the contours
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Ignore small contours
                continue
                
            motion_detected = True
            
            # Compute the bounding box for the contour and draw it in the ROI
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        if motion_detected:
            # Add text showing motion detection
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, "Motion Detected in Rectangle", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, timestamp, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("Rectangle Noise Detection", frame)
        
        # Update background occasionally
        if time.time() % 10 < 0.1:  # Update roughly every 10 seconds
            _, background = cap.read()
            background_roi = background[rect_top:rect_bottom, rect_left:rect_right]
            background_gray = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
        
        # If 'ESC' is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    return