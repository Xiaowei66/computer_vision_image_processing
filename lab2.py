# Template for lab02 task 3

import cv2
import math
import numpy as np
import sys

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.31
            print(params["contrast_threshold"])
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

# Rotate an image
def rotate(image, angle):
    
    # get the image height, width
    (h,w) = image.shape[:2]
    #calculate the center of the image
    center = (w/2,h/2)
    # 1.0 mean, the shape is preserved. Other value scales the image by the value provided.
    scale = 1.0
    # Perform the counter clockwise rotation holding at the center
    M = cv2.getRotationMatrix2D(center, angle, scale)  
    rotated_img = cv2.warpAffine(image,M,(w,h))
    
    # return the matrix of rotated image
    return rotated_img

# 
def task3(image,angle):

    t3_image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #copy the original gray image matrix
    ori_gary = np.copy(t3_image_gray)

    # get the rotated imgae matrix
    rotaed_gray = rotate(t3_image_gray,angle)

    #get the sift
    ori_sift = SiftDetector.get_detector(ori_gary,None)
    rotated_sift = SiftDetector.get_detector(rotaed_gray,None)

    # get the SIFT features

    ori_kp, ori_des = ori_sift.detectAndCompute(ori_gary,None)

    rotated_kp, rotated_des = rotated_sift.detectAndCompute(rotaed_gray,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ori_des,rotated_des,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    task3_ans_image = cv2.drawMatchesKnn(ori_gary,ori_kp,rotaed_gray,rotated_kp,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    filename = f"task3_{angle}.jpg"
    cv2.imwrite(filename,task3_ans_image)

if __name__ == '__main__':

    image = cv2.imread("road_sign.jpg")

    task3(image,0)
    task3(image,-45)
    task3(image,-90)