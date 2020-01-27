import imutils
import cv2
import numpy as np
import dlib
import time

cap = cv2.VideoCapture("./videos/test_2_25.MP4") # test video.mp4
cap.set(1, 1) # starting with 1st video frame

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

FRAME_HEIGHT = 400
cropped_margin = 15
alfa_1 = 45
alfa_2 = 255
filter_eye_margin = -5 #margin for cutted eye

def new_method():
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
        width = (landmarks.part(45).x - landmarks.part(42).x) + 2*cropped_margin
        height = (landmarks.part(46).y - landmarks.part(44).y) + 2*cropped_margin

        coeff_margin_x = cropped_margin / width
        coeff_margin_y = cropped_margin / height

        coeff_x, coeff_y = [], []
        for i in range(42, 48):
            coeff_x = np.append(coeff_x, ((landmarks.part(i).x - landmarks.part(42).x) / width))
            coeff_y = np.append(coeff_y, ((landmarks.part(i).y - landmarks.part(44).y) / height))
        #===========================================================================================#
        
        # left        
        #eye for threshold detection if there is not enough circle points 
        y_axis_start = landmarks.part(44).y-cropped_margin
        y_axis_end = landmarks.part(46).y+cropped_margin
        x_axis_start = landmarks.part(42).x-cropped_margin
        x_axis_end = landmarks.part(45).x+cropped_margin
        cropped_eye_gauss = gray[y_axis_start:y_axis_end, x_axis_start:x_axis_end]

        cropped_eye_gauss = cv2.resize(cropped_eye_gauss, (400, 200), interpolation=cv2.INTER_CUBIC)
        gauss = cv2.GaussianBlur(cropped_eye_gauss, (5, 5), 0)

        threshold_gauss = first_threshold(gauss)

        # margin coefficient
        margin_x = cropped_eye_gauss.shape[1] * coeff_margin_x
        margin_y = cropped_eye_gauss.shape[0] * coeff_margin_y

        x, y = [], []

        for i in range(0, 6):
            x = np.append(x, int((cropped_eye_gauss.shape[1] * coeff_x[i])) + int(margin_x))
            y = np.append(y, int((cropped_eye_gauss.shape[0] * coeff_y[i])) + int(margin_y))
        
        # left eye for display 
        image = image[landmarks.part(44).y-cropped_margin:landmarks.part(46).y+cropped_margin,landmarks.part(42).x-cropped_margin:landmarks.part(45).x+cropped_margin]
        image = cv2.resize(image, (400, 200), interpolation=cv2.INTER_CUBIC)

        tab_x = []
        tab_y = []

        for i in range(0, 6):
            tab_x = np.append(tab_x, int((image.shape[1] * coeff_x[i])) + int(margin_x))
            tab_y = np.append(tab_y, int((image.shape[0] * coeff_y[i])) + int(margin_y))

        sum_1, sum_2, sum_3 = 0, 0, 0
        sum_pix_1, sum_pix_2, sum_pix_3 = 0, 0, 0

        for i in range(0, threshold_gauss.shape[1]):     # x
            y_1 = ((tab_y[1] - tab_y[0])*(i - tab_x[0]))/(tab_x[1]-tab_x[0]) + tab_y[0] - filter_eye_margin  #done
            y_2 = ((tab_y[2] - tab_y[1])*(i - tab_x[1]))/(tab_x[2]-tab_x[1]) + tab_y[1] - filter_eye_margin   #done
            y_3 = ((tab_y[3] - tab_y[2])*(i - tab_x[2]))/(tab_x[3]-tab_x[2]) + tab_y[2] - filter_eye_margin  #done
            y_4 = ((tab_y[3] - tab_y[4])*(i - tab_x[4]))/(tab_x[3]-tab_x[4]) + tab_y[4] + filter_eye_margin   #done
            y_5 = ((tab_y[4] - tab_y[5])*(i - tab_x[5]))/(tab_x[4]-tab_x[5]) + tab_y[5] + filter_eye_margin  #done
            y_6 = ((tab_y[5] - tab_y[0])*(i - tab_x[0]))/(tab_x[5]-tab_x[0]) + tab_y[0] + filter_eye_margin  #done
            for j in range(0, threshold_gauss.shape[0]):  # y
                if j > y_1 and j < y_6 and i < x[1]:  # RIGHT
                    if threshold_gauss[j, i] == 0:
                        sum_1 += 1
                    sum_pix_1 += 1
                if j > y_2 and j < y_5 and i > x[1] and i < x[2]:  # CENTER
                    if threshold_gauss[j, i] == 0:
                        sum_2 += 1
                    sum_pix_2 +=1
                if j > y_3 and j < y_4 and i > x[2]:  # LEFT
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

        threshold_gauss = delete_pixels(threshold_gauss, present_state, x, y, gauss)     #MATLAB
        # hough transform after black pixels filtration
        if(present_state == "center"):
            circles = cv2.HoughCircles(threshold_gauss, cv2.HOUGH_GRADIENT, 2.0, 2, param1=40, param2=25, minRadius=50, maxRadius=60)
        else:
            circles = cv2.HoughCircles(threshold_gauss, cv2.HOUGH_GRADIENT, 2.0, 2, param1=40, param2=25, minRadius=30, maxRadius=50)

        #threshold, x_est, y_est;       
        if circles is None or len(circles[0, :, 0]) < 1: #if there is less points then 5 -> threshold
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


def first_threshold(gauss):
    _, threshold_gauss = cv2.threshold(gauss, alfa_1, alfa_2, cv2.THRESH_BINARY)
    
    b_pix = 0
    sum_pix = 0

    for k in range(0, 100):
        b_pix = 0
        sum_pix = 0
        for i in range(len(threshold_gauss[0, :])):     # x
            for j in range(len(threshold_gauss[:, 0])):  # y
                if threshold_gauss[j, i] == 0:
                    b_pix += 1
                sum_pix += 1
        if (b_pix/sum_pix > 0.2 and b_pix/sum_pix < 0.3):
            return threshold_gauss
        elif (b_pix/sum_pix < 0.2):
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 + k, 255, cv2.THRESH_BINARY)
        elif (b_pix/sum_pix > 0.3):
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 - k, 255, cv2.THRESH_BINARY)         

def delete_pixels_left(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[3]) + 15
    left_border = int(x_points[2]) - 30
    top_border = int(y_points[2]) - 15
    bottom_border = int(y_points[4]) + 15

    pixels = 0
    for i in range(0, 400):     # x
        for j in range(0, 200):  # y
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
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 - k, 255, cv2.THRESH_BINARY)

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
    for i in range(0, 400):     # x
        for j in range(0, 200):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1
    return threshold_gauss

def delete_pixels_right(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[1]) + 30
    left_border = int(x_points[0]) - 15
    top_border = int(y_points[1]) - 15
    bottom_border = int(y_points[5]) + 15

    pixels = 0
    for i in range(0, 400):     # x
        for j in range(0, 200):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1

    threshold_gauss[:, 0:left_border] = 255
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

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
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 - k, 255, cv2.THRESH_BINARY)

    b_pix = 0
    sum_pix = 0
    for i in range(left_border, right_border):     # x
        for j in range(top_border, bottom_border):  # y
            if threshold_gauss[j, i] == 0:
                b_pix += 1
            sum_pix += 1
        
    pixels = 0
    for i in range(0, 400):     # x
        for j in range(0, 200):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1
    
    return threshold_gauss

def delete_pixels_center(threshold_gauss, x_points, y_points, gauss):
    right_border = int(x_points[2]) + 20
    left_border = int(x_points[1]) - 5
    top_border = int(y_points[2]) - 15
    bottom_border = int(y_points[4]) + 10

    pixels = 0
    for i in range(0, 400):     # x
        for j in range(0, 200):  # y
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
        if (b_pix/sum_pix < 0.8):
            break
        else: 
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 - k, 255, cv2.THRESH_BINARY) 


    pixels = 0
    for i in range(left_border, right_border):     # x
        for j in range(top_border, bottom_border):  # y
            if threshold_gauss[j, i] == 0:
                pixels += 1

    threshold_gauss[:, 0:left_border] = 255   #[y, x]
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border:, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    # pixels = 0
    # for i in range(0, 400):     # x
    #     for j in range(0, 200):  # y
    #         if threshold_gauss[j, i] == 0:
    #             pixels += 1
    return threshold_gauss

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
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 - k, 255, cv2.THRESH_BINARY)

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
            _, threshold_gauss = cv2.threshold(gauss, alfa_1 - k, 255, cv2.THRESH_BINARY)

    threshold_gauss[:, 0:left_border] = 255   #[y, x]
    threshold_gauss[:, right_border:threshold_gauss.shape[1]] = 255
    threshold_gauss[0:top_border, :] = 255
    threshold_gauss[bottom_border:threshold_gauss.shape[0], :] = 255

    return threshold_gauss

def delete_pixels(threshold_gauss, present_state, x_points, y_points, gauss):
    if present_state == "right":
        threshold_gauss = delete_pixels_right(threshold_gauss, x_points, y_points, gauss)
    elif present_state == "center": 
        threshold_gauss = delete_pixels_center(threshold_gauss, x_points, y_points, gauss)
    elif present_state == "left":
        threshold_gauss = delete_pixels_left(threshold_gauss, x_points, y_points, gauss)
    # elif present_state == "bottom":
    #     threshold_gauss = delete_pixels_bottom(threshold_gauss, x_points, y_points, gauss)
    # elif present_state == "top":
    #     threshold_gauss = delete_pixels_top(threshold_gauss, x_points, y_points, gauss)

    return threshold_gauss

def hough_for_canny(circles):
    sum_X = np.sum(circles[0, :, 0])
    sum_Y = np.sum(circles[0, :, 1])

    x_est = int(sum_X / len(circles[0, :, 0]))
    y_est = int(sum_Y / len(circles[0, :, 1]))

    return x_est, y_est

def drawing_lines(threshold_gauss, coeff_x, coeff_y, margin_x, margin_y ):
    for i in range(0, 5):
        x_1 = int((threshold_gauss.shape[1] * coeff_x[i])) + int(margin_x)
        y_1 = int((threshold_gauss.shape[0] * coeff_y[i])) + int(margin_y)
        x_2 = int((threshold_gauss.shape[1] * coeff_x[i+1])) + int(margin_x)
        y_2 = int((threshold_gauss.shape[0] * coeff_y[i+1])) + int(margin_y)
        cv2.line(threshold_gauss, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
    x_1 = int((threshold_gauss.shape[1] * coeff_x[0])) + int(margin_x)
    y_1 = int((threshold_gauss.shape[0] * coeff_y[0])) + int(margin_y)
    x_2 = int((threshold_gauss.shape[1] * coeff_x[5])) + int(margin_x)
    y_2 = int((threshold_gauss.shape[0] * coeff_y[5])) + int(margin_y)
    cv2.line(threshold_gauss, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)

    return threshold_gauss

#hough transform using 2 value image and estimate two points using their sum
def hough_transform_2_pixels(threshold):
    #x_est
    sum_hough_tranfsorm = np.sum(threshold, axis=0)
    x0, = np.where(sum_hough_tranfsorm[:] == min(sum_hough_tranfsorm))
    x_est = min(x0)
    #y_est
    sum_hough_tranfsorm = np.sum(threshold, axis=1)
    y0, = np.where(sum_hough_tranfsorm[:] == min(sum_hough_tranfsorm))
    y_est = min(y0)

    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
    
    return threshold, x_est, y_est

new_method()
       
