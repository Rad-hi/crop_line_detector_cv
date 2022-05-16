"""
 CV approach for crop line detection and heading angle error estimation.
 -----------------------------------------------------------------------------
 Author: Radhi SGHAIER: https://github.com/Rad-hi
 Heavily based on: https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/
 -----------------------------------------------------------------------------
 Date: 16-05-2022 (16th of May, 2022)
 -----------------------------------------------------------------------------
 License: Do whatever you want with the code ...
          If this was ever useful to you, and we happened to meet on 
          the street, I'll appreciate a cup of dark coffee, no sugar please.
 -----------------------------------------------------------------------------
"""

import cv2
import numpy as np
import math

# Viz options' bitmasks
NO_VIZ                     = 0x00
DRAW_LANE_AREA_MASK        = 0x01 << 0 # Viz the Crop lane mask before and after warping
DRAW_WARPED_LANES          = 0x01 << 1 # Viz the polynomially interpolated lines on the warped frame
DRAW_WINDOWS_ON_FRAME      = 0x01 << 2 # Viz the windows as they're drawn
DRAW_SLIDING_WINDOW_RESULT = 0x01 << 3 # Viz the sliding windows on the ROI
DRAW_ANGLE_ERROR_ON_IMAGE  = 0x01 << 4 # Viz the heading error on the image
DRAW_CENTER_ESTIMATIONS    = 0x01 << 5 # Viz the heading center estimation on the final frame
DRAW_FINAL_RESULT          = 0x01 << 6 # Viz the final result

class cropLineDetector():
    '''Crop line detector class'''

    # Colors "BGR"
    RED    = [0, 0, 255]
    BLUE   = [255, 0, 0]
    GREEN  = [0, 255, 0]
    BLACK  = [0, 0, 0]
    WHITE  = [255, 255, 255]
    YELLOW = [255, 255, 0]

    # This color isn't only used for vizualisation, 
    # check self._find_left_right_intersections() before changing (although it doesn't affect it working)
    CROP_LANE_COLOR = GREEN

    # Percentage of the height/width that define the ROI
    D_L_B_X = 0.33 # Down left bound on the X axis
    D_R_B_X = 0.68
    U_L_B_X = 0.41
    U_R_B_X = 0.62

    # The ROI can't Go beyond 40% (bottom to top) of the image
    UPPER_B_Y = 0.6

    # Sliding window parameters
    WINDOW_RADIUS = 15 # The window's half width
    WINDOW_WIDTH = 2*WINDOW_RADIUS
    NUMBER_OF_WINDOWS = 10
    MIN_PIXELS_IN_WINDOW = 25
    WINDOW_HEGHT_LIMIT_IN_FRAME = 1.0 - UPPER_B_Y


    def __init__(self, original_frame, poly_deg=1, viz_options=NO_VIZ):
        """
        @param orginal_frame : The first frame to enter the detector
        @param poly_deg      : The degree of the polynomial we'll try to fit within the points that shall make up the line
        @param viz_options   : An 8-bit bitwise flag for the various visualization options (check bitmasks above)
        """

        # Used as a template for the copied of the image
        self._zeros_image = np.zeros_like(original_frame)

        # The vizualisations bitwise flag
        self._viz_mask = viz_options

        self._height, self._width = original_frame.shape[:2]
        self._poly_degree = poly_deg

        # This is the ROI on the real feed
        self._real_roi_points = [ # Trapezoid
            (int(self.D_L_B_X*self._width), int(self._height)), # Down left point
            (int(self.D_R_B_X*self._width), int(self._height)), # Down right point 
            (int(self.U_R_B_X*self._width), int(self.UPPER_B_Y*self._height)), # Upper right point
            (int(self.U_L_B_X*self._width), int(self.UPPER_B_Y*self._height)), # Upper left point
        ]
        
        # This is the ROI in the warped space (how we want the ROI to be warped)
        self._desired_roi_points = [ # A Retangle in the warped space
            (int(self.U_L_B_X*self._width), int(self.UPPER_B_Y*self._height)), # Upper left point
            (int(self.U_R_B_X*self._width), int(self.UPPER_B_Y*self._height)), # Upper right point
            (int(self.U_R_B_X*self._width), int(self._height)), # Down right point
            (int(self.U_L_B_X*self._width), int(self._height)), # Down left point 
        ]

        # Go from real world perspective to bird's eye perspective 
        self._transformation_matrix = cv2.getPerspectiveTransform(np.float32(self._real_roi_points), 
                                                                  np.float32(self._desired_roi_points))

        # Go from bird's eye perspective back to real world perspective
        self._inv_transformation_matrix = cv2.getPerspectiveTransform(np.float32(self._desired_roi_points), 
                                                                      np.float32(self._real_roi_points))

        # Data containers
        self._prev_left_x     = []
        self._prev_left_y     = []
        self._prev_right_x    = []
        self._prev_right_y    = []
        self._prev_left_x_2   = []
        self._prev_left_y_2   = [] 
        self._prev_right_x_2  = []
        self._prev_right_y_2  = []
        self._prev_left_fit2  = []
        self._prev_right_fit2 = []
        self._left_fit_x      = []
        self._right_fit_x     = []
        self._ploty           = []

        # The heading angle error
        self._error_angle = 0.0


    def get_heading_angle_error(self, current_frame):
        """Main task"""
        copy_of_current_frame = np.copy(current_frame)

        canny_img = self._apply_canny_filter(current_frame, 80, 180)
        roi = self._get_region_of_interest(canny_img)

        zeros_img = np.copy(self._zeros_image)
        warped_img = cv2.warpPerspective(roi, self._transformation_matrix, (self._width, self._height), zeros_img)

        hist_of_img = self._calculate_historgam(warped_img)
        left_peak, right_peak = self._histogram_peaks(hist_of_img)

        left_fit, right_fit = self._get_crop_line_indicies_sliding_window(warped_img, left_peak, right_peak, poly_degree=self._poly_degree)
        self._get_crop_line_previous_window(warped_img, left_fit, right_fit, poly_degree=self._poly_degree)
        
        result = self._get_crop_lines_estimation()
        
        if self._viz_mask & DRAW_FINAL_RESULT:
            final = self._overlay_crop_lines(result, copy_of_current_frame)
            cv2.imshow("Final result", final)
        
        return self._error_angle

    def _get_region_of_interest(self, image):
        """Mask the ROI of an image using pts"""
        polygons = np.array([self._real_roi_points])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, self.WHITE)
        return cv2.bitwise_and(image, mask)

    @staticmethod
    def _apply_canny_filter(image, min_thresh, max_thresh):
        """Extract edges from the image"""
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        return cv2.Canny(blur_img, threshold1=min_thresh, threshold2=max_thresh)

    def _draw_line(self, lines):
        """Draw a line on an image"""
        line_img = np.copy(self._zeros_image)
        for x0, y0, x1, y1 in lines:
            cv2.line(line_img, (x0, y0), (x1, y1), self.BLUE, 5)
        return line_img

    @staticmethod
    def _calculate_historgam(image):
        return np.sum(image[int(image.shape[0]/2):,:], axis=0)

    @staticmethod
    def _histogram_peaks(hist):
        """Get peaks on the left and right parts of the hist"""
        midpoint = int(hist.shape[0]/2)
        # Left peak, Right peak
        return np.argmax(hist[:midpoint]),np.argmax(hist[midpoint:]) + midpoint

    @staticmethod
    def _evaluate_polynomial(y, coefs, degree):
        """Evaluates a polynomial at a certain array-like series of points"""
        return sum([x*(y**(degree-i)) for i, x in enumerate(coefs)])

    def _get_crop_line_indicies_sliding_window(self, image, left_peak, right_peak, poly_degree=1):
        """Get the frame specific windows indicies"""
        
        # Set the height of the sliding windows
        window_height = int(self.WINDOW_HEGHT_LIMIT_IN_FRAME*(self._height/self.NUMBER_OF_WINDOWS))		

        # Find the x and y coordinates of all white pixels 
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])	
            
        # Store the pixel indices for the left and right lane lines
        left_crop_row_inds = []
        right_crop_row_inds = []
            
        # Current positions for pixel indices for each window,
        # which we will continue to update
        current_left_x = left_peak
        current_right_x = right_peak

        for window in range(self.NUMBER_OF_WINDOWS):
            
            # Identify window boundaries in x and y (and right and left)
            win_y_low        = self._height - (window + 1) * window_height
            win_y_high       = self._height - window * window_height
            win_x_left_low   = current_left_x  - self.WINDOW_RADIUS
            win_x_left_high  = current_left_x  + self.WINDOW_RADIUS
            win_x_right_low  = current_right_x - self.WINDOW_RADIUS
            win_x_right_high = current_right_x + self.WINDOW_RADIUS
            
            if self._viz_mask & DRAW_WINDOWS_ON_FRAME:
                cv2.rectangle(image, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), self.WHITE, 1)
                cv2.rectangle(image, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), self.WHITE, 1)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) 
                            & (nonzeroy <  win_y_high) 
                            & (nonzerox >= win_x_left_low) 
                            & (nonzerox <  win_x_left_high)).nonzero()[0]

            good_right_inds = ((nonzeroy >= win_y_low)
                             & (nonzeroy < win_y_high)
                             & (nonzerox >= win_x_right_low) 
                             & (nonzerox <  win_x_right_high)).nonzero()[0]
                                                            
            # Append these indices to the lists
            left_crop_row_inds.append(good_left_inds)
            right_crop_row_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on mean position
            if len(good_left_inds) > self.MIN_PIXELS_IN_WINDOW:
                current_left_x = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.MIN_PIXELS_IN_WINDOW:        
                current_right_x = int(np.mean(nonzerox[good_right_inds]))
                    
        # Concatenate the arrays of indices
        left_crop_row_inds = np.concatenate(left_crop_row_inds)
        right_crop_row_inds = np.concatenate(right_crop_row_inds)

        # Extract the pixel coordinates for the left and right lane lines
        left_x  = nonzerox[left_crop_row_inds]
        left_y  = nonzeroy[left_crop_row_inds]
        right_x = nonzerox[right_crop_row_inds] 
        right_y = nonzeroy[right_crop_row_inds]

        # Make sure we have nonzero pixels		
        if len(left_x)==0 or len(left_y)==0 or len(right_x)==0 or len(right_y)==0:
            left_x  = self._prev_left_x
            left_y  = self._prev_left_y
            right_x = self._prev_right_x
            right_y = self._prev_right_y

        # Holder for the polynomial
        left_fit = []
        right_fit = []

        # Get the left and right fits
        self._ploty = np.linspace(0, self._height-1, self._height)
        left_fit    = np.polyfit(left_y, left_x, poly_degree)
        right_fit   = np.polyfit(right_y, right_x, poly_degree)
        self._left_fit_x  = self._evaluate_polynomial(self._ploty, left_fit, poly_degree)
        self._right_fit_x = self._evaluate_polynomial(self._ploty, right_fit, poly_degree)
        
        # Keep these results for in case the next frame is empty
        self._prev_left_x  = left_x
        self._prev_left_y  = left_y 
        self._prev_right_x = right_x
        self._prev_right_y = right_y

        if self._viz_mask & DRAW_WARPED_LANES:
            # Generate an image to visualize the result
            out_img = np.dstack((image, image, (image))) * 255
            
            # Add color to the left line pixels and right line pixels
            out_img[nonzeroy[left_crop_row_inds], nonzerox[left_crop_row_inds]] = self.BLUE
            out_img[nonzeroy[right_crop_row_inds], nonzerox[right_crop_row_inds]] = self.RED

            cv2.putText(image, "Sliding window result", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, self.YELLOW, 1)
            cv2.imshow("Warped & Colored Slides", cv2.add(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), out_img))
        return left_fit, right_fit

    def _get_crop_line_previous_window(self, warped_image, left_fit, right_fit, poly_degree=1):
        """Interpolate based on the previous windows"""

        # Find the x and y coordinates of all the nonzero 
        # (i.e. white) pixels in the frame.			
        nonzero = warped_image.nonzero()  
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Isolate the needed pixel indicies
        left_interpolation = self._evaluate_polynomial(nonzeroy, left_fit, poly_degree)
        left_crop_row_inds  = ((nonzerox > (left_interpolation - self.WINDOW_RADIUS))
                             & (nonzerox < (left_interpolation + self.WINDOW_RADIUS)))
        
        right_interpolation = self._evaluate_polynomial(nonzeroy, right_fit, poly_degree)
        right_crop_row_inds = ((nonzerox > (right_interpolation - self.WINDOW_RADIUS))
                             & (nonzerox < (right_interpolation +  self.WINDOW_RADIUS)))
        
        # Get the left and right lane line pixel locations	
        left_x  = nonzerox[left_crop_row_inds]
        left_y  = nonzeroy[left_crop_row_inds]
        right_x = nonzerox[right_crop_row_inds]
        right_y = nonzeroy[right_crop_row_inds]	

        # Make sure we have nonzero pixels		
        if len(left_x)==0 or len(left_y)==0 or len(right_x)==0 or len(right_y)==0:
            left_x  = self._prev_left_x_2
            left_y  = self._prev_left_y_2
            right_x = self._prev_right_x_2
            right_y = self._prev_right_y_2

        
        # Get the left and right fits
        ploty = np.linspace(0, self._height-1, self._height)
        left_fit = np.polyfit(left_y, left_x, poly_degree)
        right_fit = np.polyfit(right_y, right_x, poly_degree)
        self._left_fit_x  = self._evaluate_polynomial(self._ploty, left_fit, poly_degree)
        self._right_fit_x = self._evaluate_polynomial(self._ploty, right_fit, poly_degree)

        # Add the latest polynomial coefficients		
        self._prev_left_fit2.append(left_fit)
        self._prev_right_fit2.append(right_fit)

        # Calculate the moving average	
        if len(self._prev_left_fit2) > 10:
            self._prev_left_fit2.pop(0)
            self._prev_right_fit2.pop(0)
            left_fit  = sum(self._prev_left_fit2) / len(self._prev_left_fit2)
            right_fit = sum(self._prev_right_fit2) / len(self._prev_right_fit2)
            
        # Hold these values for if we encounter an empty frame
        self._prev_left_x_2, self._prev_right_x_2 = (left_x, right_x)
        self._prev_left_y_2, self._prev_right_y_2 = (left_y, right_y)
            
        if self._viz_mask & DRAW_SLIDING_WINDOW_RESULT:
            # Generate images to draw on
            out_img = np.dstack((warped_image, warped_image, (warped_image)))*255
            window_img = np.zeros_like(out_img)
                
            # Add color to the left and right line pixels
            out_img[nonzeroy[left_crop_row_inds], nonzerox[left_crop_row_inds]] = self.BLUE
            out_img[nonzeroy[right_crop_row_inds], nonzerox[right_crop_row_inds]] = self.RED
            
            # Create a polygon to show the search window area, and recast 
            # the x and y points into a usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([self._left_fit_x-self.WINDOW_RADIUS, self._ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self._left_fit_x+self.WINDOW_RADIUS, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([self._right_fit_x-self.WINDOW_RADIUS, self._ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self._right_fit_x+self.WINDOW_RADIUS, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, left_line_pts.astype(int), self.WHITE)
            cv2.fillPoly(window_img, right_line_pts.astype(int), self.WHITE)

            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
            cv2.putText(result, "Interpolated crop lines on warped ROI", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, self.WHITE, 1)
            cv2.imshow("Sliding window result", result)

    @staticmethod
    def _find_left_right_intersections(line, line_height, area_color, background_color=BLACK):
        """Find the left and right "intersections" within a image line with a fictive area"""
        l = []
        r = []
        first = False
        for i, pixel in enumerate(line):
            if (pixel==area_color).all() and not first:
                first = True
                l = [i, line_height] # First point of the area
            elif (pixel==background_color).all() and first:
                first = False
                r = [i, line_height] # Last point of the area
                break
        
        return l, r
    
    def _get_heading_error(self, bin_crop_line_img):
        """Finds the center of the crop lane area at a certain height"""

        LOW_LINE_HEIGHT = int(0.9*self._height)
        UP_LINE_HEIGHT = int(0.65*self._height)
        low_evaluation_line = bin_crop_line_img[LOW_LINE_HEIGHT]
        up_evaluation_line = bin_crop_line_img[UP_LINE_HEIGHT]
        
        # Holder for the left and right points of the evaluation lines' intersections with the "crop lane"
        left_point = {0:[], 1:[]}
        right_point = {0:[], 1:[]}

        # Find left and right boundaries of the evaluation area "trapezoidal"
        left_point[0], right_point[0] = self._find_left_right_intersections(low_evaluation_line, LOW_LINE_HEIGHT, self.CROP_LANE_COLOR, self.BLACK)
        left_point[1], right_point[1] = self._find_left_right_intersections(up_evaluation_line, UP_LINE_HEIGHT, self.CROP_LANE_COLOR, self.BLACK)
                
        # We have points
        if left_point[0] and left_point[1] and right_point[0] and right_point[1]:
            
            center_point = (int((right_point[0][0] - left_point[0][0])/2) + left_point[0][0], LOW_LINE_HEIGHT)
            
            # Find the goal orientation (middle of the up line)
            goal_arrow_point = (int((right_point[1][0] - left_point[1][0])/2) + left_point[1][0], UP_LINE_HEIGHT)
            
            # Find our actual orientation
            length = int(math.dist(center_point, goal_arrow_point))
            heading_arrow_point = (center_point[0], center_point[1] - length)
            
            # Calculate the angle between the two lines ( cos(theta) = adj/hyp --> theta = acos(adj/hyp) )
            cos_angel = float(math.dist(center_point, heading_arrow_point)
                            /math.dist(center_point, goal_arrow_point))
            self._error_angle = np.arccos(cos_angel)

            if self._viz_mask & DRAW_CENTER_ESTIMATIONS:
                # Draw evaluation line
                EVAL_LINE_WINDOW_RADIUS = 20 # How much the evaluation line extends from both ends of the crop lane region
                left_side_line = (left_point[0][0] - EVAL_LINE_WINDOW_RADIUS, LOW_LINE_HEIGHT)
                right_side_line = (right_point[0][0] + EVAL_LINE_WINDOW_RADIUS, LOW_LINE_HEIGHT)
                cv2.line(bin_crop_line_img, left_side_line, right_side_line, self.WHITE, 2)

                # Draw intersection points
                RADIUS = 4
                cv2.circle(bin_crop_line_img, left_point[0], RADIUS, self.BLUE, 2)
                cv2.circle(bin_crop_line_img, right_point[0], RADIUS, self.BLUE, 2)

                # Draw the center point
                cv2.circle(bin_crop_line_img, center_point, RADIUS, self.BLACK, 2)

                # Draw the goal and heading arrows
                cv2.arrowedLine(bin_crop_line_img, center_point, goal_arrow_point, self.BLACK, 1)
                cv2.arrowedLine(bin_crop_line_img, center_point, heading_arrow_point, self.RED, 2)

    def _get_crop_lines_estimation(self):
        # We need a black image to hold the mask (but that can hold colors --> convert it to 3D) 
        color_warp = np.copy(self._zeros_image)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self._left_fit_x, self._ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self._right_fit_x, self._ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, pts.astype(int), self.CROP_LANE_COLOR)
        cropped_color_warp = self._get_region_of_interest(color_warp)

        if self._viz_mask & DRAW_LANE_AREA_MASK:
            cv2.imshow("Before warping", cropped_color_warp)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        ready_to_overlay = cv2.warpPerspective(cropped_color_warp, 
                                               self._inv_transformation_matrix,
                                               (self._width, self._height))

        if self._viz_mask & DRAW_LANE_AREA_MASK:
            cv2.imshow("After warping", ready_to_overlay)

        self._get_heading_error(ready_to_overlay)
        
        return ready_to_overlay

    def _overlay_crop_lines(self, crop_lines_mask_img, original_img):
        """Overlays the interpolated crop lines on the original image"""
        if DRAW_ANGLE_ERROR_ON_IMAGE:
            cv2.putText(original_img, "Angle error: "+str(self._error_angle)[:5], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, self.RED, 1)
        # Combine the result with the original image
        return cv2.addWeighted(original_img, 1, crop_lines_mask_img, 0.5, 0)