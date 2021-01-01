#!/usr/bin/env python3

import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import cv2
import numpy as np
from skimage import morphology


npz_index = 0
def save_npz(img, boxes, classes, segm):

    if segm:
        DATASET_DIR="../../dataset/sim_mask"
        with makedirs(DATASET_DIR):
            np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
    else:
        DATASET_DIR="../../dataset/sim_obs"
        with makedirs(DATASET_DIR):
            np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
 

count = [0, 0, 0, 0, 0]
def clean_segmented_image(seg_img):

# https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python

    y_seg_img = seg_img.shape[0]
    x_seg_img = seg_img.shape[1]

    perc_threshold = [0.1, 0.005, 0.003, 0.0005, 0.01]
    k = [7, 4, 2, 7, 7]
    low_color = []
    high_color = []
    n_samples = 300
    global count

    # # background

    low_color.append(np.array([0,100,255]))
    high_color.append(np.array([255,255,255]))

    # # #  ducks
    low_color.append(np.array([0, 142, 0]))
    high_color.append(np.array([255, 143, 255]))

    # #  cones

    low_color.append(np.array([0, 140, 0]))
    high_color.append(np.array([255, 141, 255]))

    # # # truck

    low_color.append(np.array([0, 0, 0]))
    high_color.append(np.array([255, 50, 200]))

    # bus

    low_color.append(np.array([23,100,0]))
    high_color.append(np.array([24,255,255]))

    crop_image = []
    boxes = []
    classes = []
    crop_boxes = []

    imgHSV = cv2.cvtColor(seg_img, cv2.COLOR_RGB2HSV)

    for i in range(5):
        if count[i] < n_samples:
            mask = cv2.inRange(imgHSV, low_color[i], high_color[i])

            res = cv2.bitwise_and(seg_img, seg_img, mask=mask)

            ret, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)  

            kernel = np.ones((k[i], k[i]), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            cleaned = morphology.remove_small_objects(opening, min_size=65, connectivity=3)

            perc_object = np.count_nonzero(cleaned.flatten())/len(cleaned.flatten())

            if perc_object > perc_threshold[i]:
  
                gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    x_center = int(x + w/2)
                    y_center = int(y + h/2)

                    if x_center > 112:
                        x_inf = x_center - 112
                    else:
                        x_inf = 0

                    if (x_inf + 224) < x_seg_img:
                        x_sup = x_inf + 224
                    else:
                        x_sup = x_seg_img

                    if y_center > 112:
                        y_inf = y_center - 112
                    else:
                        y_inf = 0

                    if (y_inf + 224) < y_seg_img:
                        y_sup = y_inf + 224
                    else:
                        y_sup = y_seg_img

                    if ((x_sup - x_inf) == 224 and (y_sup - y_inf) == 224):
                        crop_img = cleaned[y_inf:y_sup, x_inf:x_sup] 
                        gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                       
                        # cv2.imshow("im_2colors", crop_img)
                        # cv2.waitKey(100)

                        crop_image.append(crop_img)
                        crop_boxes.append([x_inf, y_inf, x_sup, y_sup])

                        contours_crop, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                        boxes_list = []
                        classes_list = []   

                        for contour_crop in contours_crop:
                            x, y, w, h  = cv2.boundingRect(contour_crop)
                            boxes_list.append([x, y, x+w, y+h])
                            classes_list.append(i)
                        
                        boxes.append(np.array(boxes_list))
                        classes.append(np.array(classes_list))    
                        count[i] += 1

    
    if crop_image is None:
        return None
    else:
        return crop_image, crop_boxes, boxes, classes 

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 10

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # boxes, classes = clean_segmented_image(segmented_obs)
        crop_image, crop_boxes, boxes, classes = clean_segmented_image(segmented_obs)

        for i in range(len(crop_image)):

            # segmented
            save_npz(crop_image[i], boxes[i], classes[i], segm=True)

            crop_img_not_segm = obs[crop_boxes[i][1]:crop_boxes[i][3], crop_boxes[i][0]:crop_boxes[i][2]] 
            save_npz(crop_img_not_segm, boxes[i], classes[i], segm=False)

            npz_index += 1

        print(count)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break