import cv2
import numpy as np

def spot_diff(frame1, frame2):
    # Convert frames to grayscale if they are not already
    if len(frame1[1].shape) == 3:
        frame1_gray = cv2.cvtColor(frame1[1], cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1[1]
        
    if len(frame2[1].shape) == 3:
        frame2_gray = cv2.cvtColor(frame2[1], cv2.COLOR_BGR2GRAY)
    else:
        frame2_gray = frame2[1]
    
    # Calculate absolute difference between frames
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    
    # Apply threshold to highlight differences
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # If there are significant differences
    if len(significant_contours) > 0:
        # Draw rectangles around the differences
        for contour in significant_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2[1], (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Display the frames with differences highlighted
        cv2.imshow("Before", frame1[1])
        cv2.imshow("After", frame2[1])
        cv2.imshow("Differences", thresh)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return 1  # Differences found
    else:
        return 0  # No significant differences