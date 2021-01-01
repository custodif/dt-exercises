import numpy as np
import cv2

DATASET_DIR="../../dataset"


for i in range(500):
        with np.load(f"{DATASET_DIR}/{i}.npz") as data:

                im = data[f"arr_{0}"]
                bx = data[f"arr_{1}"]
                cl = data[f"arr_{2}"]
            
        # cv2.imshow(f"loaded_{i}", im)
        # cv2.waitKey(5000)
        # print(bx)
        for a in range(bx.shape[0]):
            crop_img_not_segm = im[bx[a][1]:bx[a][3], bx[a][0]:bx[a][2]] 
            # if cl[a] == 0:
            #     cv2.imshow(f"crop_{i}_{a}_background", crop_img_not_segm)
            # if cl[a] == 1:
            #     cv2.imshow(f"crop_{i}_{a}_duckies", crop_img_not_segm)
            if cl[a] == 2:
                cv2.imshow(f"crop_{i}_{a}_cones", crop_img_not_segm)        
            # if cl[a] == 3:
            #     cv2.imshow(f"crop_{i}_{a}_truck", crop_img_not_segm)     
        #     if cl[a] == 4:
        #         cv2.imshow(f"crop_{i}_{a}_bus", crop_img_not_segm)                                             
        cv2.waitKey(1000)

