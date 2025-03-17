import cv2
import cv2 as cv
import cv2 as cv
import cv2
import numpy as np
from time import time
import logging
import math



class LaneFollower:
    def get_steering_angle(self, frame, lane_lines):

        height, width, _ = frame.shape
        
        x_offset, y_offset = 0, int(height / 2)  # Default offsets

        if len(lane_lines) == 2:
            # Extract lane line coordinates
            left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]
            right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]

            # Calculate slopes in radians
            slope_l = math.atan2(left_y2 - left_y1, left_x2 - left_x1)
            slope_r = math.atan2(right_y2 - right_y1, right_x2 - right_x1)
            slope_ldeg = int(slope_l * 180.0 / math.pi)
            steering_angle_left = slope_ldeg
            slope_rdeg = int(slope_r * 180.0 / math.pi)
            steering_angle_right = slope_rdeg
            # Determine offset based on slopes
            if left_x2>right_x2:
                if abs(steering_angle_left)<= abs(steering_angle_right):
                    x_offset = left_x2 - left_x1
                    y_offset = int(height/2)
                elif abs(steering_angle_right) > abs(steering_angle_right):
                    x_offset = right_x2 -right_x1
                    y_offset = int(height/2)
            else:
                mid = int(width/2)
                x_offset = (left_x2 + right_x2) / 2 - mid
                y_offset = int(height / 2)

        elif len(lane_lines) == 1:
            # Single lane line detected
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
            y_offset = int(height / 2)

        elif len(lane_lines) == 0:
            # No lane lines detected
            x_offset = 0
            y_offset = int(height / 2)


        # Calculate the steering angle
        alfa = 0.65
        angle_to_mid_radian = alfa * getattr(self, 'angle', 0) + (1 - alfa) * math.atan(x_offset / y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
        # Final steering angle
        steering_angle = angle_to_mid_deg + 90
        self.angle = angle_to_mid_radian  # Save angle for smoothing

        return steering_angle


    def display_heading_line(self, frame, steering_angle, line_color=(255,0,0), line_width=4):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape
        steering_angle_radian = steering_angle / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 1.75)
        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

        return heading_image

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv.addWeighted(initial_img, α, img, β, λ)


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv.HoughLinesP(cropped_edges, rho, angle, min_threshold,np.array([]), minLineLength=20,maxLineGap=200)  #maxLineGap=15

    return line_segments


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down
    if slope !=0 and not math.isinf(slope): # fixed error for the case that the slope is infinite 
    
    # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    else :
        x1 =0
        x2 =0
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def region_of_interest(edges):
    #height, width = edges.shape
    height = edges.shape[0]
    width = edges.shape[1]
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (width*0.2 , height *0.6),
        (width*0.6 ,height*0.6),
        (width, height), # right side 
        (0, height),
    ]], np.int32)
    """polygon = np.array([[
        (0, height ),
        (width, height ),
        (width, 130),
        (0, 130),
    ]], np.int32)"""
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    return cropped_edges
'''
def region_of_interest(edges, image):
    img = np.zeros_like(edges)
    imshape = image.shape
    lower_left = [0+imshape[1]/10, imshape[0]]
    lower_right = [imshape[1]-imshape[1]/10, imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 4, imshape[0] / 2 + imshape[0] / 6]
    top_right = [imshape[1] / 2 + imshape[1] / 4, imshape[0] / 2 + imshape[0] / 6]
        
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    if len(edges.shape) > 2:
        mask_color_ignore = (255, ) * edges.shape[2]
    else:
        mask_color_ignore = 255
    cv2.fillPoly(img, vertices, mask_color_ignore)
    cropped_edges = cv2.bitwise_and(edges, img)
    # io.imshow(img)
    return cropped_edges

 '''