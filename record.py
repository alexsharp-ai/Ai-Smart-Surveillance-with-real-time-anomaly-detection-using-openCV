import cv2
import numpy as np
import time
from datetime import datetime
import os

def record():
    # Create recordings directory if it doesn't exist
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    cap = cv2.VideoCapture(0)  # Use default webcam
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # If fps is not available, default to 20
        fps = 20
    
    # Define the codec and create VideoWriter object
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/recording_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    
    # Set recording duration (in seconds)
    record_duration = 30
    start_time = time.time()
    
    print(f"Recording started. Duration: {record_duration} seconds")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add timestamp to the frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, "RECORDING", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, timestamp, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write the frame to the output file
        out.write(frame)
        
        # Display the frame
        cv2.imshow("Recording", frame)
        
        # Check if recording duration is reached
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= record_duration:
            break
        
        # Display remaining time
        remaining = int(record_duration - elapsed_time)
        cv2.putText(frame, f"Remaining: {remaining}s", (frame_width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # If 'ESC' is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Recording saved to {filename}")
    return