import imutils
import cv2
import numpy as np
import dlib
import time
from utils.constants import FRAME_HEIGHT, FRAME_WIDTH, CROPPED_MARGIN, ALFA_MIN, ALFA_MAX, FILTER_EYE_MARGIN
from utils.threshold import first_threshold, second_threshold

cap = cv2.VideoCapture("./videos/test_2_25.MP4") # test video.mp4
cap.set(1, 1) # starting with 1st video frame

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Method to track LEFT eye movement
def main():
    while True:
        ret, image = cap.read()
        # image = np.rot90(np.rot90(np.rot90(image)))

        startTime = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

        #===========================================================================================#
        # distance from point to point coefficients
        width = (landmarks.part(45).x - landmarks.part(42).x) + 2*CROPPED_MARGIN
        height = (landmarks.part(46).y - landmarks.part(44).y) + 2*CROPPED_MARGIN

        coeff_margin_x = CROPPED_MARGIN / width
        coeff_margin_y = CROPPED_MARGIN / height

        coeff_x, coeff_y = [], []
        for i in range(42, 48):
            coeff_x = np.append(coeff_x, ((landmarks.part(i).x - landmarks.part(42).x) / width))
            coeff_y = np.append(coeff_y, ((landmarks.part(i).y - landmarks.part(44).y) / height))
        #===========================================================================================#
        
        # take frame fragment containing left eye
        y_axis_start = landmarks.part(44).y-CROPPED_MARGIN
        y_axis_end = landmarks.part(46).y+CROPPED_MARGIN
        x_axis_start = landmarks.part(42).x-CROPPED_MARGIN
        x_axis_end = landmarks.part(45).x+CROPPED_MARGIN
        cropped_eye_gauss = gray[y_axis_start:y_axis_end, x_axis_start:x_axis_end]

        # interpolation
        cropped_eye_gauss = cv2.resize(cropped_eye_gauss, (FRAME_HEIGHT, FRAME_WIDTH), interpolation=cv2.INTER_CUBIC)
        
        #filtration
        gauss = cv2.GaussianBlur(cropped_eye_gauss, (5, 5), 0)

        # first threshold needed to estimate where pupil can possibly be
        threshold_gauss = first_threshold(gauss)

        # margin coefficient
        margin_x = FRAME_HEIGHT * coeff_margin_x
        margin_y = FRAME_WIDTH * coeff_margin_y

        tab_x, tab_y = [], []

        for i in range(0, 6):
            tab_x = np.append(tab_x, int((FRAME_HEIGHT * coeff_x[i])) + int(margin_x))
            tab_y = np.append(tab_y, int((FRAME_WIDTH * coeff_y[i])) + int(margin_y))
        
        # left eye for display 
        image = image[landmarks.part(44).y-CROPPED_MARGIN:landmarks.part(46).y+CROPPED_MARGIN,landmarks.part(42).x-CROPPED_MARGIN:landmarks.part(45).x+CROPPED_MARGIN]
        image = cv2.resize(image, (FRAME_HEIGHT, FRAME_WIDTH), interpolation=cv2.INTER_CUBIC)

        sum_1, sum_2, sum_3 = 0, 0, 0
        sum_pix_1, sum_pix_2, sum_pix_3 = 0, 0, 0

        for i in range(0, FRAME_HEIGHT):     # x
            y_1 = ((tab_y[1] - tab_y[0])*(i - tab_x[0]))/(tab_x[1]-tab_x[0]) + tab_y[0] - FILTER_EYE_MARGIN  #done
            y_2 = ((tab_y[2] - tab_y[1])*(i - tab_x[1]))/(tab_x[2]-tab_x[1]) + tab_y[1] - FILTER_EYE_MARGIN   #done
            y_3 = ((tab_y[3] - tab_y[2])*(i - tab_x[2]))/(tab_x[3]-tab_x[2]) + tab_y[2] - FILTER_EYE_MARGIN  #done
            y_4 = ((tab_y[3] - tab_y[4])*(i - tab_x[4]))/(tab_x[3]-tab_x[4]) + tab_y[4] + FILTER_EYE_MARGIN   #done
            y_5 = ((tab_y[4] - tab_y[5])*(i - tab_x[5]))/(tab_x[4]-tab_x[5]) + tab_y[5] + FILTER_EYE_MARGIN  #done
            y_6 = ((tab_y[5] - tab_y[0])*(i - tab_x[0]))/(tab_x[5]-tab_x[0]) + tab_y[0] + FILTER_EYE_MARGIN  #done
            for j in range(0, FRAME_WIDTH):  # y
                if j > y_1 and j < y_6 and i < tab_x[1]:  # RIGHT
                    if threshold_gauss[j, i] == 0:
                        sum_1 += 1
                    sum_pix_1 += 1
                if j > y_2 and j < y_5 and i > tab_x[1] and i < tab_x[2]:  # CENTER
                    if threshold_gauss[j, i] == 0:
                        sum_2 += 1
                    sum_pix_2 +=1
                if j > y_3 and j < y_4 and i > tab_x[2]:  # LEFT
                    if threshold_gauss[j, i] == 0:
                        sum_3 += 1
                    sum_pix_3 +=1

        # # present_state = "center"
        # if (np.all(threshold_gauss[int(y[5]+(y[4]-y[5])/2)-5:int(y[5]+(y[4]-y[5])/2), int(x[5])-20: int(x[4])+20] == 255)):
        #     print("top")
        #     present_state = "top"
        if sum_1/sum_pix_1 > 0.7:
            print("right")
            present_state = "right"
        elif sum_2/sum_pix_2 > 0.8:
            # if (y[5] - y[1] <= 62 and y[4] - y[2] <= 62):
            #     print("bottom")
            #     present_state = "bottom"
            # else:
            print("center")
            present_state = "center"
        elif sum_3/sum_pix_3 > 0.7:
            print("left")
            present_state = "left"
        else:
            print(present_state)

        # second threshold and deleting pixels
        threshold_gauss = second_threshold(threshold_gauss, tab_x, tab_y, gauss, present_state)

        # hough transform after black pixels filtration
        if(present_state == "center"):
            circles = cv2.HoughCircles(threshold_gauss, cv2.HOUGH_GRADIENT, 2.0, 2, param1=40, param2=25, minRadius=50, maxRadius=60)
        else:
            circles = cv2.HoughCircles(threshold_gauss, cv2.HOUGH_GRADIENT, 2.0, 2, param1=40, param2=25, minRadius=30, maxRadius=50)
   
        if circles is None or len(circles[0, :, 0]) < 1:
            threshold_gauss, x_est, y_est = hough_transform_2_pixels(threshold_gauss)
        else:
            x_est, y_est = hough_for_canny(circles)
            threshold_gauss = cv2.cvtColor(threshold_gauss, cv2.COLOR_GRAY2RGB)

            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)
                cv2.circle(threshold_gauss, (i[0], i[1]), 2, (0, 0, 255), 2)

        #===========drawing lines between landmarks==============#
        threshold_gauss = drawing_lines(threshold_gauss, coeff_x, coeff_y, margin_x, margin_y)
        image = drawing_lines(image, coeff_x, coeff_y, margin_x, margin_y)
        
        cv2.circle(threshold_gauss, (x_est, y_est), 2, (255, 0, 0), 2)  #center point
        cv2.circle(image, (x_est, y_est), 2, (0, 255, 255), 2)  #center point

        endTime = time.time()
        print("Pogram time: ", endTime - startTime)
    
        cv2.imshow("Binary eye with points", threshold_gauss)
        cv2.imshow("Eye with points", image)

        key = cv2.waitKey(30)
        if key == 27:
            break

def hough_for_canny(circles):
    sum_X = np.sum(circles[0, :, 0])
    sum_Y = np.sum(circles[0, :, 1])

    x_est = int(sum_X / len(circles[0, :, 0]))
    y_est = int(sum_Y / len(circles[0, :, 1]))

    return x_est, y_est

#hough transform using 2 value image and estimate two points using their sum
def hough_transform_2_pixels(threshold_gauss):
    #x_est
    sum_hough_tranfsorm = np.sum(threshold_gauss, axis=0)
    x0, = np.where(sum_hough_tranfsorm[:] == min(sum_hough_tranfsorm))
    x_est = min(x0)
    #y_est
    sum_hough_tranfsorm = np.sum(threshold_gauss, axis=1)
    y0, = np.where(sum_hough_tranfsorm[:] == min(sum_hough_tranfsorm))
    y_est = min(y0)

    threshold_gauss = cv2.cvtColor(threshold_gauss, cv2.COLOR_GRAY2RGB)
    
    return threshold_gauss, x_est, y_est

def drawing_lines(threshold_gauss, coeff_x, coeff_y, margin_x, margin_y ):
    for i in range(0, 5):
        x_1 = int((FRAME_HEIGHT * coeff_x[i])) + int(margin_x)
        y_1 = int((FRAME_WIDTH * coeff_y[i])) + int(margin_y)
        x_2 = int((FRAME_HEIGHT * coeff_x[i+1])) + int(margin_x)
        y_2 = int((FRAME_WIDTH * coeff_y[i+1])) + int(margin_y)
        cv2.line(threshold_gauss, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
    x_1 = int((FRAME_HEIGHT * coeff_x[0])) + int(margin_x)
    y_1 = int((FRAME_WIDTH * coeff_y[0])) + int(margin_y)
    x_2 = int((FRAME_HEIGHT * coeff_x[5])) + int(margin_x)
    y_2 = int((FRAME_WIDTH * coeff_y[5])) + int(margin_y)
    cv2.line(threshold_gauss, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)

    return threshold_gauss

main()
       
