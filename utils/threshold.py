from utils.constants import ALFA_MIN, ALFA_MAX, FRAME_HEIGHT, FRAME_WIDTH
import cv2
import numpy as np

def first_threshold(gauss):
    _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN, ALFA_MAX, cv2.THRESH_BINARY)

    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0
        
        arr = threshold_gauss[0:len(threshold_gauss[:, 0]),0:len(threshold_gauss[0, :])]
        b_pix = np.count_nonzero(arr == 0)
        sum_pix = len(threshold_gauss[0, :])*len(threshold_gauss[:, 0])

        if (b_pix/sum_pix > 0.2 and b_pix/sum_pix < 0.3):
            return threshold_gauss
        elif (b_pix/sum_pix < 0.2):
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN + k, ALFA_MAX, cv2.THRESH_BINARY)
        elif (b_pix/sum_pix > 0.3):
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN - k, ALFA_MAX, cv2.THRESH_BINARY)         

def second_threshold(threshold_gauss, x_points, y_points, gauss, present_state):
    if present_state == "right":
        return delete_pixels_right(threshold_gauss, x_points, y_points, gauss)
    elif present_state == "center": 
        return delete_pixels_center(threshold_gauss, x_points, y_points, gauss)
    elif present_state == "left":
        return delete_pixels_left(threshold_gauss, x_points, y_points, gauss)
    
def delete_pixels_left(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[3]) + 15
    left_border = int(x_points[2]) - 30
    top_border = int(y_points[2]) - 15
    bottom_border = int(y_points[4]) + 15

    pixels = 0
    for i in range(0, FRAME_HEIGHT):     # x
        for j in range(0, FRAME_WIDTH):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1
    
    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0
        for i in range(left_border, right_border):     # x
            for j in range(top_border, bottom_border):  # y
                if threshold_gauss[j, i] == 0:
                    b_pix += 1
                sum_pix += 1
        if (b_pix/sum_pix < 0.5):
            break
        else:
            print("TEST")
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN - k, ALFA_MAX, cv2.THRESH_BINARY)

    threshold_gauss[:, 0:left_border] = 255
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    pixels = 0
    for i in range(left_border, right_border):     # x
        for j in range(top_border, bottom_border):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1
    pixels = 0
    for i in range(0, FRAME_HEIGHT):     # x
        for j in range(0, FRAME_WIDTH):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1
    return threshold_gauss

def delete_pixels_right(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[1]) + 30
    left_border = int(x_points[0]) - 15
    top_border = int(y_points[1]) - 15
    bottom_border = int(y_points[5]) + 15

    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0
        
        arr = threshold_gauss[top_border:bottom_border, left_border:right_border]
        b_pix = np.count_nonzero(arr == 0)
        sum_pix = (right_border-left_border)*(bottom_border-top_border)
        
        if (b_pix/sum_pix < 0.5):
            break
        else: 
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN - k, ALFA_MAX, cv2.THRESH_BINARY)
    
    threshold_gauss[:, 0:left_border] = 255
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    return threshold_gauss

def delete_pixels_center(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[2]) + 20
    left_border = int(x_points[1]) - 5
    top_border = int(y_points[2]) - 15
    bottom_border = int(y_points[4]) + 15
    
    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0

        arr = threshold_gauss[top_border:bottom_border, left_border:right_border]
        b_pix = np.count_nonzero(arr == 0)
        sum_pix = (right_border-left_border)*(bottom_border-top_border)
        
        if (b_pix/sum_pix < 0.8):
            break
        else: 
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN - k, ALFA_MAX, cv2.THRESH_BINARY) 

    threshold_gauss[:, 0:left_border] = 255   #[y, x]
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border:, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    return threshold_gauss

# to do - NOT USED
def delete_pixels_bottom(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[2]) + 5
    left_border = int(x_points[1]) - 5
    top_border = int(y_points[2]) - 10
    bottom_border = int(y_points[4]) + 10

    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0
        for i in range(left_border, right_border):     # x
            for j in range(top_border, bottom_border):  # y
                if threshold_gauss[j, i] == 0:
                    b_pix += 1
                sum_pix += 1
        if (b_pix/sum_pix < 0.7):
            break
        else: 
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN - k, ALFA_MAX, cv2.THRESH_BINARY)

    threshold_gauss[:, 0:left_border] = 255   #[y, x]
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    return threshold_gauss

def delete_pixels_top(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[2]) + 0
    left_border = int(x_points[1]) - 0
    top_border = int(y_points[2]) + 0
    bottom_border = int(y_points[2]) + 40

    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0
        for i in range(left_border, right_border):     # x
            for j in range(top_border, bottom_border):  # y
                if threshold_gauss[j, i] == 0:
                    b_pix += 1
                sum_pix += 1
            # if np.any(threshold_gauss[:, i] == 0):
            #     b_pix += 1
        if (b_pix/sum_pix < 0.7):
            break
        else: 
            _, threshold_gauss = cv2.threshold(gauss, ALFA_MIN - k, ALFA_MAX, cv2.THRESH_BINARY)

    threshold_gauss[:, 0:left_border] = 255   #[y, x]
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    return threshold_gauss
