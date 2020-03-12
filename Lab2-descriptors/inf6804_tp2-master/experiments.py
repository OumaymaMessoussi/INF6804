import cv2
import numpy as np
import matplotlib.pyplot as plt

import sgm
import distances


def get_recall(disparity, gt, max_disp):
    """
    computes the recall of the disparity map.
    :param disparity: disparity image.
    :param gt: path to ground-truth image.
    :param max_disp: maximum disparity.
    :return: rate of correct predictions.
    """
    gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    if gt.shape[0] > 800:
        gt = cv2.resize(gt, (gt.shape[1]//2, gt.shape[0]//2))
    gt = np.int16(gt / 255.0 * float(max_disp))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(max_disp))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    
    fig = plt.figure(figsize=(200, 200))
    
    x = fig.add_subplot(1,2, 1)
    x.set_title('gt')
    plt.imshow(gt, cmap='gray')

    y = fig.add_subplot(1,2, 2)
    y.set_title('disparity map')
    plt.imshow(disparity, cmap='gray')

    plt.show()
    
    return float(correct) / gt.size


def experiments(descriptor, left, right, gt_name, descriptor_size):
    
    print(left)
    print(right)
    print(gt_name)

    if descriptor == 'brief':
        print('Running BRIEF..\n')
        distance = distances.hamming_distance
    else:
        print('Running HOG..\n')
        distance = distances.l2_distance
        
    num_orientation = 8
    num_elements = 128
    max_disp = 64

    disparity_map = sgm.sgm(left, right, descriptor, distance, descriptor_size, num_elements, num_orientation, max_disp)
    cv2.imwrite('left_disp_map.png', disparity_map)

    print('\nEvaluating left disparity map...')
    recall = get_recall(disparity_map, gt_name, max_disp)
    print('\tRecall = {:.2f}%\n'.format(recall * 100.0))

if __name__ == '__main__':
    experiments()
    experiments_rotated()
