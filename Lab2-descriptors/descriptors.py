import sys
import time as t

import numpy as np
from PIL import ImageOps
from skimage.feature import hog, BRIEF
import matplotlib.pyplot as plt
import cv2


def apply_hog(left_im, right_im, descriptor_size, num_orientations):
    """
    computes HOG descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: number of pixels in a hog cell.
    :param num_orientations: number of HOG orientations.
    :return: (H x W x M) array, H = height, W = width and M = num_orientations, of type np.float32.
    """
    # TODO: apply HOG descriptor on left and right images.
    
    i_min = descriptor_size//2
    
    left = np.zeros((left_im.shape[0], left_im.shape[1], num_orientations), dtype=np.float32)
    right = np.zeros((right_im.shape[0], right_im.shape[1], num_orientations), dtype=np.float32)
    
    # Computing feature descriptors for the left image
    
    for i in range(i_min, left_im.shape[0]-i_min):
        for j in range(i_min, left_im.shape[1]-i_min):
            
            ROI = left_im[(i-i_min):(i+i_min+1), (j-i_min):(j+i_min+1)]
            left[i][j] = hog(ROI, num_orientations, pixels_per_cell=(descriptor_size, descriptor_size), 
                             cells_per_block=(1, 1), feature_vector=True)

    # Computing feature descriptors for the right image

    for i in range(i_min, right_im.shape[0]-i_min):
        for j in range(i_min, right_im.shape[1]-i_min):
            
            ROI = right_im[(i-i_min):(i+i_min+1), (j-i_min):(j+i_min+1)]
            right[i][j] = hog(ROI, num_orientations, pixels_per_cell=(descriptor_size, descriptor_size), 
                              cells_per_block=(1, 1), feature_vector=True)
                 
    print(left.shape)
    
    return left, right


def apply_brief(imgL, imgR, descriptor_size, num_elements):
    """
    computes BRIEF descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: size of window of the BRIEF descriptor.
    :param num_elements: length of the feature vector.
    :return: (H x W) array, H = height and W = width, of type np.int64
    """
    # TODO: apply BRIEF descriptor on both images. You will have to convert the BRIEF feature vector to a int64.

    # Image pixel indices to use as "key points"
    
    indicesL = np.zeros(((imgL.shape[0])*(imgL.shape[1]), 2), dtype=np.int64)
    indicesR = np.zeros(((imgR.shape[0])*(imgR.shape[1]), 2), dtype=np.int64)
    
    k = 0
    for i in range(0, imgL.shape[0]):
        for j in range(0, imgL.shape[1]):
            indicesL[k,0], indicesL[k,1] = i, j
            k += 1
    k = 0
    for i in range(0, imgR.shape[0]):
        for j in range(0, imgR.shape[1]):
            indicesR[k,0], indicesR[k,1] = i, j
            k += 1
            
    # BRIEF descriptor

    extractor = BRIEF(descriptor_size=num_elements, patch_size=descriptor_size)
    
    extractor.extract(imgL, indicesL)
    descriptorL = extractor.descriptors
    extractor.extract(imgR, indicesR)
    descriptorR = extractor.descriptors
    
    # Converting descriptor values to integer (int64)
  
    descL = np.ndarray((descriptorL.shape[0],1), dtype=np.int64)
    descR = np.ndarray((descriptorR.shape[0],1), dtype=np.int64)

    for i in range(descriptorL.shape[0]):
        descL[i] = np.int64(descriptorL[i].dot(2**np.arange(descriptorL[i].size)[::-1]))
        
    for i in range(descriptorR.shape[0]):
        descR[i] = np.int64(descriptorR[i].dot(2**np.arange(descriptorR[i].size)[::-1]))
    
    # Reshaping the descriptor and adding padding to match the image shape
    
    descriptorL = descL.reshape((imgL.shape[0]-(descriptor_size-2), imgL.shape[1]-(descriptor_size-2)))
    descriptorR = descR.reshape((imgR.shape[0]-(descriptor_size-2), imgR.shape[1]-(descriptor_size-2)))
    
    delta_w = imgL.shape[1] - descriptorL.shape[1]
    delta_h = imgL.shape[0] - descriptorL.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    descriptorL = cv2.copyMakeBorder(descriptorL, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    descriptorR = cv2.copyMakeBorder(descriptorR, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
    plt.imshow(descriptorL, cmap='gray')
    plt.show()
    
    return descriptorL, descriptorR