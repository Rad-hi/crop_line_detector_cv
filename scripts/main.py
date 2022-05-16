from cropLineDetector import *

import cv2
import numpy as np

# Insert a video path
INPUT_FILE_PATH = "/home/Radhi/Desktop/crop_line_detector/images/crops.mp4"

# The interpolated polynomial's degree
POLY_DEGREE  = 1

def main():
    cap = cv2.VideoCapture(INPUT_FILE_PATH)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    detector = None
    while detector is None:    
        ret, image = cap.read()
        if not ret:
            continue
        else:
            detector = cropLineDetector(original_frame=image,
                                        poly_deg=POLY_DEGREE,
                                        viz_options=DRAW_FINAL_RESULT|
                                                    DRAW_CENTER_ESTIMATIONS)

    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        # The purpose of this whole class is to determine the heading angle error
        heading_angle_error = detector.get_heading_angle_error(image)

        # Correct using the determined angle!
        print(heading_angle_error)

        # This waits for a key press to advance for debugging purposes
        if cv2.waitKey(0) == ord('q'):
            break
    else:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()