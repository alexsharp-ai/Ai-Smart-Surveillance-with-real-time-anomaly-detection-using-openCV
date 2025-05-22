import cv2
import numpy as np
import time
from datetime import datetime
import os

def in_out():
    # Create directory for storing entry/exit images if it doesn't exist
    if not os.path.exists('entries'):
        os.makedirs('entries')
    
    cap = cv2.VideoCapture("rtsp://B48mfZuY:1ilmri3ObV99IADP@192.168.2.12:554/live/ch0")  # Use RTSP stream
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Error: Couldn't load face cascade classifier.")
        return
    
    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # Parameters
    entry_direction = "right"  # People enter from right side of frame
    min_contour_area = 5000   # Minimum contour area to be considered a person
    
    # Variables to track people
    people_entered = 0
    people_exited = 0
    active_tracks = []
    
    print("Monitoring for entries and exits...")
    
    # Wait for camera to initialize
    time.sleep(2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply background subtraction
        fg_mask = backSub.apply(frame)
        
        # Apply some morphology operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center of the contour
            center_x = x + w // 2
            
            # Draw rectangle around the contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Check for faces in this region to confirm it's a person
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            if len(faces) > 0:
                # Person detected
                person_id = f"{datetime.now().timestamp()}"
                
                # Determine if entering or exiting based on position
                if entry_direction == "right":
                    # If moving from right to left (x decreasing)
                    if center_x < frame.shape[1] // 2:
                        if person_id not in active_tracks:
                            people_entered += 1
                            active_tracks.append(person_id)
                            
                            # Save image of the person entering
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"entries/entry_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"Person entered. Image saved to {filename}")
                    else:
                        if person_id not in active_tracks:
                            people_exited += 1
                            active_tracks.append(person_id)
                            
                            # Save image of the person exiting
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"entries/exit_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"Person exited. Image saved to {filename}")
                
                # Draw face rectangle
                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(frame[y:y+h, x:x+w], (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        
        # Clean up old tracks
        if len(active_tracks) > 10:
            active_tracks = active_tracks[-10:]
            
        # Add counters to the frame
        cv2.putText(frame, f"Entered: {people_entered}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Exited: {people_exited}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw dividing line
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Entry/Exit Detection", frame)
        
        # If 'ESC' is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Final count - Entered: {people_entered}, Exited: {people_exited}")
    return
