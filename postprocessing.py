"""!
"""
import cv2
import numpy as np
import imutils
from skimage.draw import line


def postprocessing(img=None,
                   mask=None,
                   radius=5,
                   block_size=17,
                   constant=-35.5,
                   threshold=-1.0,
                   ecc_thres=0.8):
    """!@brief

    @return
        A list of a dictionary {'x': , 'y': }.
    """
    # Assertions
    assert img is not None
    assert block_size > 1
    assert (block_size % 2) == 1

    # Array of detections
    det = []

    # Apply threshold technique
    # =========================
    if threshold >= 0.0:
        # Normalize input image
        img32 = cv2.normalize(src=img,
                              dst=None,
                              alpha=0,
                              beta=1.0,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_32F)

        # Apply thresholding to resulting map
        binary_img = cv2.threshold(src=img32,
                                   thresh=threshold,
                                   maxval=255,
                                   type=cv2.THRESH_BINARY)[1].astype(np.uint8)
    else:
        # Normalize input image
        img8 = cv2.normalize(src=img,
                             dst=None,
                             alpha=0,
                             beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8UC1)

        # Apply adaptive threshold to the image
        binary_img = cv2.adaptiveThreshold(src=img8,
                                           maxValue=255,
                                           adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thresholdType=cv2.THRESH_BINARY,
                                           blockSize=block_size,
                                           C=constant)

    # Apply mask
    # ==========
    if mask is not None:
        binary_img = cv2.bitwise_and(binary_img, binary_img, mask=mask)

    # # Show resulting image after fusion of feature maps
    # import matplotlib.pyplot as plt
    # plt.matshow(binary_img, cmap='gray')
    # plt.show()

    #**************************************************************************

    # # Get points from each binary object in the resulting image
    # # =========================================================
    # n_labels, img_labeled = cv2.connectedComponents(image=binary_img,
    #                                                 connectivity=8,
    #                                                 ltype=cv2.CV_16U)

    # # ===================================================
    # # # @DEBUG
    # # # Show resulting image after fusion of feature maps
    # # import matplotlib.pyplot as plt
    # # plt.matshow(img_labeled)
    # # plt.show()
    # # ===================================================

    # # Normalize input image
    # img = cv2.normalize(src=img,
    #                     dst=None,
    #                     alpha=0,
    #                     beta=255,
    #                     norm_type=cv2.NORM_MINMAX,
    #                     dtype=cv2.CV_8UC1)

    # # Get center points
    # center = _get_point_label(img, img_labeled, n_labels)

    # =============================================
    # # @DEBUG
    # # Draw a cross in center points
    # centers_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for c in center:
    #     cv2.drawMarker(img=centers_img,
    #                    position=c,
    #                    color=(255, 0, 0),
    #                    markerType=cv2.MARKER_CROSS,
    #                    markerSize=4,
    #                    thickness=1)
    # # Displaying the image
    # cv2.imshow("Centers before eccentricity threshold", centers_img)
    # cv2.waitKey(0)
    # =============================================

    # # Loop for each binary object in the image
    # # ========================================
    # for i in range(1, n_labels):
    #     # Get measure value
    #     ecc = _get_eccentricity(img, center[i], radius)

    #     # Apply an eccentricity threshold
    #     if ecc >= ecc_thres:
    #         # Add centroid values in det array
    #         dict_cnt = {'x': center[i][0], 'y': center[i][1]}
    #         det.append(dict_cnt)

    # return det

    #**************************************************************************

    # Get points from each binary object in the resulting image
    # =========================================================
    cnts = cv2.findContours(binary_img,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Array of detections
    det = []

    # Loop over the contours
    for c in cnts:
        # Compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] > 0.0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = c[0][0][0]
            cY = c[0][0][1]

        # Add centroid value in det array
        dict_cnt = {'x': cX, 'y': cY}
        det.append(dict_cnt)

    return det


def _get_point_label(img=None,
                     img_labeled=None,
                     n_labels=0):
    """!@brief
    """
    # Assertions
    assert img is not None
    assert img_labeled is not None
    assert n_labels > 0

    # Initialize arrays
    centers = [None] * n_labels
    max_values = np.zeros(n_labels, dtype=np.int8)

    # Loop for all pixels in the image
    #   Obs: shape[0] = high or rows
    #        shape[1] = width or cols
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):

            l = img_labeled[x][y]
            pixel = img[x][y]

            # Check background
            if l > 0:
                if pixel > max_values[l]:
                    max_values[l] = pixel
                    centers[l] = (y, x)

    return centers


def _get_eccentricity(img=None,
                      center=None,
                      r=5):
    """!@brief
    """
    # Assertions
    assert img is not None
    assert center is not None
    assert r >= 0
    r = int(round(r))

    # Initialize variables
    aux_vec = []
    acum_line = np.zeros(int(2*r+1))

    # ======================
    # Get eccentricity lines
    # ======================

    # Vertical line |
    # --------------------------------------
    pt1 = (int(center[0]), int(center[1]-r))
    pt2 = (int(center[0]), int(center[1]+r))

    # @DEBUG
    #   Draw vertical line
    # test = np.zeros_like(img)
    # cv2.line(test, pt1, pt2, (255, 255, 255))

    # Returns a list of point coordinates
    # that corresponds to the line between
    # points pt1 and pt2
    line_profile = list(zip(*line(*pt1, *pt2)))

    # Get line profile
    for l in line_profile:
        # Extrapolation up and down limits
        if l[1] < 0 or l[1] >= img.shape[0]:
            aux_vec.append(0.0)
        else:
            aux_vec.append(img[l[1]][l[0]])

    # Acumulate lines and clean
    # auxiliar array
    acum_line = np.add(acum_line, aux_vec)
    aux_vec = []

    # Horizontal line -
    # --------------------------------------
    pt1 = (int(center[0]-r), int(center[1]))
    pt2 = (int(center[0]+r), int(center[1]))

    # @DEBUG
    #   Draw horizontal line
    # cv2.line(test, pt1, pt2, (255, 255, 255))

    # Get line coordinates
    line_profile = list(zip(*line(*pt1, *pt2)))

    # Get line profile
    for l in line_profile:
        # Extrapolation left and right limits
        if l[0] < 0 or l[0] >= img.shape[1]:
            aux_vec.append(0.0)
        else:
            aux_vec.append(img[l[1]][l[0]])

    # Acumulate lines and clean
    # auxiliar array
    acum_line = np.add(acum_line, aux_vec)
    aux_vec = []

    # Diagonal line \
    # ----------------------------------------
    pt1 = (int(center[0]-r), int(center[1]-r))
    pt2 = (int(center[0]+r), int(center[1]+r))

    # @DEBUG
    #   Draw diagonal line
    # cv2.line(test, pt1, pt2, (255, 255, 255))

    # Get line coordinates
    line_profile = list(zip(*line(*pt1, *pt2)))

    # Get line profile
    for l in line_profile:
        # Extrapolation left and right limits
        if (l[0] < 0 or              # left  limit
            l[1] < 0 or              # up    limit
            l[0] >= img.shape[1] or  # right limit
            l[1] >= img.shape[0]):   # down  limit
            aux_vec.append(0.0)
        else:
            aux_vec.append(img[l[1]][l[0]])

    # Acumulate lines and clean
    # auxiliar array
    acum_line = np.add(acum_line, aux_vec)
    aux_vec = []

    # Diagonal line /
    # ----------------------------------------
    pt1 = (int(center[0]+r), int(center[1]-r))
    pt2 = (int(center[0]-r), int(center[1]+r))

    # @DEBUG
    #   Draw diagonal line
    # cv2.line(test, pt1, pt2, (255, 255, 255))

    # Get line coordinates
    line_profile = list(zip(*line(*pt1, *pt2)))

    # Get line profile
    for l in line_profile:
        # Extrapolation left and right limits
        if (l[0] < 0 or              # left  limit
            l[1] < 0 or              # up    limit
            l[0] >= img.shape[1] or  # right limit
            l[1] >= img.shape[0]):   # down  limit
            aux_vec.append(0.0)
        else:
            aux_vec.append(img[l[1]][l[0]])

    # Acumulate lines and clean
    # auxiliar array
    acum_line = np.add(acum_line, aux_vec)
    aux_vec = []

    # @DEBUG
    #   Show image with line draws
    # print(acum_line)
    # cv2.imshow("lines", test)
    # cv2.waitKey(0)

    # Compute the average of line profiles
    avg_line = np.sum(acum_line) / len(acum_line)

    # Compute measure SQR_DIFF_NORM
    # =============================

    # Define correlation
    max_correl = 0.0

    # Estimate a Gaussian kernel for different sigma values
    # @See https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#Mat%20getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype)
    ksize = 2*r+1
    sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

    n_kernels = 6
    sigma_step = 0.25 * r / n_kernels
    max_sigma = sigma + n_kernels * sigma_step

    while sigma < max_sigma:
        # Create Gaussian kernel vector
        gauss = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
        kernel = np.transpose(gauss)[0]

        # Compute average values for kernel array
        avg_kernel = np.sum(kernel) / len(kernel)

        # Declare accum variables
        sum_nominator = 0.0
        sum_left_denom = 0.0
        sum_right_denom = 0.0

        # Compute statistical values for each kernel
        for i in range(len(kernel)):
            # Expected values
            E_x = acum_line[i] - avg_line
            E_y = kernel[i] - avg_kernel

            # Nominator
            sum_nominator += E_x * E_y

            # Denominator
            sum_left_denom  += E_x * E_x
            sum_right_denom += E_y * E_y

        # Pearson Correlation Coefficient
        correl = sum_nominator / np.sqrt(sum_left_denom * sum_right_denom)

        # Get max correlation value
        if correl > max_correl:
            max_correl = correl

        sigma += sigma_step

    # =====================================================
    # Weigthing by the top of distribution (local maximum)
    #   OUTPUT:
    #         max_correl x local_max
    #         ----------------------
    #         n_lines x max_pixel_val
    # =====================================================
    output = (max_correl * acum_line[r]) / (4 * 255)

    return output
