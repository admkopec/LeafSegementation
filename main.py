# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import os
import cv2


def jaccard(x, y):
    lenIntersection = np.count_nonzero(x & y)
    lenX = np.count_nonzero(x)
    lenY = np.count_nonzero(y)
    return lenIntersection / (lenX + lenY - lenIntersection)


def dice(x, y):
    lenIntersection = np.count_nonzero(x & y)
    lenX = np.count_nonzero(x)
    lenY = np.count_nonzero(y)
    return 2. * lenIntersection / (lenX + lenY)

def detect_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create the mask for the brown color
    mask_brown = cv2.inRange(hsv, (8, 68, 45), (45, 255, 200))
    # Create the mask for the yellow and green colours in the leaf
    mask_yellow_green = cv2.inRange(hsv, (10, 40, 50), (95, 255, 255))
    # Create the mask for the dark green colours in the leaf
    mask_green = cv2.inRange(hsv, (30, 30, 35), (95, 255, 255))
    # Combine the above masks
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_yellow_green, mask_green), mask_brown)
    # Create a kernel for morphology close in the shape of a circle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # V1:
    # Perform a series of erosions, dilations and morphology close operations
    mask = cv2.dilate(mask, kernel, iterations=1)
    #mask = cv2.erode(mask, kernel, iterations=1)
    #mask = cv2.dilate(mask, kernel, iterations=6)
    #mask = cv2.erode(mask, kernel, iterations=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    #mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, None, iterations=1)
    # V2:
    # perform a series of erosions and dilations
    #mask = cv2.dilate(mask, None, iterations=6)
    #mask = cv2.erode(mask, None, iterations=7)
    #mask = cv2.dilate(mask, None, iterations=2)
    #mask = cv2.erode(mask, None, iterations=2)
    # Return the resulting Mask
    return mask


def true_mask(segmented):
    grayscale = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    segmentedMask = cv2.inRange(grayscale, np.array(2), np.array(255))
    segmentedMask = cv2.erode(segmentedMask, None, iterations=2)
    segmentedMask = cv2.dilate(segmentedMask, None, iterations=1)
    return segmentedMask


diceCoefficients = []
ious = []

for (_, _, filenames) in os.walk('color/'):
    for filename in filenames:
        image = cv2.imread('color/'+filename, cv2.IMREAD_COLOR)
        segmented = cv2.imread('segmented/'+filename.rsplit('.', 1)[0]+'_final_masked.JPG', cv2.IMREAD_COLOR)

        mask = detect_leaf(image)
        segmentedMask = true_mask(segmented)
        # Bitwise-AND mask and original image
        masked = cv2.bitwise_and(image, image, mask=mask)

        cv2.imwrite('my_segmented/'+filename, mask)

        coeff = dice(segmentedMask, mask)
        #print(coeff)
        diceCoefficients.append(coeff)

        iou = jaccard(segmentedMask, mask)
        #print(iou)
        ious.append(iou)

        if iou < 0.9 or coeff < 0.9:
            cv2.imshow('Mask', mask)
            cv2.imshow('Segmented Mask', segmentedMask)

            cv2.imshow('Image', image)
            cv2.imshow('Masked', masked)
            cv2.imshow('Segmented', segmented)
            cv2.imshow('Segmented Masked', cv2.bitwise_and(segmented, segmented, mask=segmentedMask))
            print(iou)
            print(coeff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    break

print(np.mean(diceCoefficients))
print(np.max(diceCoefficients))
print(np.min(diceCoefficients))
print('------')
print(np.mean(ious))
print(np.max(ious))
print(np.min(ious))
