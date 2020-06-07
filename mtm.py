"""!
"""

import cv2
import imutils
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def _get_measures(gt, det, r):
    """!@brief

    @note One ground truth point can be associated to
    more than one predicted point.
    """
    # Assertions
    gt = gt[gt['frame'] == 1]
    det = det[det['frame'] == 1]

    # Prepare data
    gt['count'] = 'FN'
    det['count'] = 'FP'

    # Find matchings and accumulate corresponding values
    # @TODO
    #       Find a faster way to perform that.
    for i_det, r_det in det.iterrows():
        # Iterate in ground truth array
        for i_gt, r_gt in gt.iterrows():

            dist = np.sqrt((r_gt['x']-r_det['x'])**2 + (r_gt['y']-r_det['y'])**2)

            if dist <= r:
                gt['count'].at[i_gt] = 'TP'
                det['count'].at[i_det] = 'TP'
                break

    # ===============================================
    #                Get countings
    # ===============================================
    try:
        TP = gt.groupby('count').size().at['TP']
    except:
        TP = 0

    try:
        FN = gt.groupby('count').size().at['FN']
    except:
        FN = 0

    try:
        FP = det.groupby('count').size().at['FP']
    except:
        FP = 0

    # Compute final measures
    P = TP / (TP+FP)
    R = TP / (TP+FN)

    if P > 0 and R > 0:
        F1 = (2*P*R) / (P+R)
    else:
        F1 = 0

    return P, R, F1


def get_best_features(features=None,
                      threshold=0.9,
                      retained_value=0.1,
                      radius=5,
                      template=None,
                      mask_img=None,
                      gt=None):
    """!@brief

    @return
        List of best feature indexes.
    """
    # Assertions
    assert features is not None
    assert template is not None
    assert gt is not None

    # Empty measures array
    v_P = np.array([])
    v_R = np.array([])
    v_F = np.array([])

    # Apply TM on all the feature images
    # ==================================
    for i in range(features.shape[2]):

        # import timeit
        # start = timeit.timeit()
        # print('Processing frame', i)

        feature_img = cv2.normalize(src=features[:, :, i],
                                    dst=None,
                                    alpha=0,
                                    beta=255,
                                    norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)

        # Create template image
        tmp = feature_img[template[1]:template[3], 
                          template[0]:template[2]] # [y0:y1, x0:x1]

        # Apply template matching
        tm_map = cv2.matchTemplate(feature_img, tmp, cv2.TM_CCOEFF_NORMED)

        # Adjust resulting map by adding borders
        w, h = tmp.shape[::-1]
        left_right_border = int((w-1)/2)
        top_down_border = int((h-1)/2)
        tm_map = cv2.copyMakeBorder(src=tm_map,
                                    top=top_down_border,
                                    bottom=top_down_border,
                                    left=left_right_border,
                                    right=left_right_border,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=0)

        # Apply thresholding to resulting map
        binary = cv2.threshold(tm_map, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Apply mask to the resulting map image
        binary = cv2.bitwise_and(binary, binary, mask=mask_img)

        # Get points from each binary object in the resulting image
        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            dict_cnt = {'frame': 1, 'x': cX, 'y': cY}
            det.append(dict_cnt)

        # Get resulting measures
        df_gt = gt[gt['frame'] == 1]
        df_det = pd.DataFrame(det)
        P, R, F = _get_measures(df_gt, df_det, radius)
        try:
            v_P = np.append(v_P, P)
            v_R = np.append(v_R, R)
            v_F = np.append(v_F, F)
        except:
            v_P = P
            v_R = R
            v_F = F

        # Get indexes according to the retained measure value
        stand_v_F = v_F / (np.sum(v_F) + np.finfo(float).eps)
        idx_sorted = np.argsort(stand_v_F)[::-1]
        cum_sorted_v_F = np.cumsum(stand_v_F[idx_sorted])
        selected_v_F = idx_sorted[np.where(cum_sorted_v_F <= retained_value)]
        
        # Add a threshold for v_F
        sub = np.where(v_F[selected_v_F] > 0.1)[0]
        sub_v_F = selected_v_F[sub]

    return sub_v_F


def template_matching(image=None,
                      templates=[],
                      scale_fusion='sum',
                      tmpl_fusion='sum'):
    """!@brief
        # ===========================================================
        #                    TEMPLATE MATCHING
        #
        # @NOTE: Other methods for TM:
        #          [ ] TM_SQDIFF          (accept a mask template)
        #          [ ] TM_SQDIFF_NORMED
        #          [ ] TM_CCORR
        #          [ ] TM_CCORR_NORMED    (accept a mask template)
        #          [ ] TM_CCOEFF
        #          [x] TM_CCOEFF_NORMED
        #
        #       Return a map between [-1, 1]
        #          -1: max anti-correlation
        #           0: not correlated
        #           1: max correlation
        #
        # @See: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_template_matching.html
        #
        # ============================================================
    """
    #Assertions
    assert image is not None
    assert len(templates) > 0

    # Normalize input image
    input_img = cv2.normalize(src=image,
                              dst=None,
                              alpha=0,
                              beta=255,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)

    # Initialize fusion map for pyramid scales
    map_fusion_tmpl = np.zeros_like(input_img, dtype=np.float32)

    # Loop for multiple templates
    # ===========================
    for tmpl_pyramid in templates:

        # Initialize fusion map for pyramid scales
        map_fusion_pyr = np.zeros_like(input_img, dtype=np.float32)

        # Loop for mutiple scales in pyramid
        # ==================================
        for tmpl in tmpl_pyramid:

            # import matplotlib.pyplot as plt
            # plt.matshow(tmpl, cmap='gray')
            # plt.show()

            # Apply Template Matching
            tm_map = cv2.matchTemplate(input_img, tmpl, cv2.TM_CCOEFF_NORMED)

            # Adjust resulting map by adding borders
            w, h = tmpl.shape[::-1]
            left_right_border = int((w-1)/2)
            top_down_border = int((h-1)/2)
            tm_map = cv2.copyMakeBorder(src=tm_map,
                                        top=top_down_border,
                                        bottom=top_down_border,
                                        left=left_right_border,
                                        right=left_right_border,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=0)

            # # Show maps for each scale in pyramid
            # plt.matshow(tm_map, cmap='gray')
            # plt.show()

            # ================================================
            #         FUSION SCALES (PYRAMID LEVELS)
            #
            # @NOTE: Another approach could be used to fusion
            #        pyramid level results
            #           - sum
            #           - max value
            #           - pixel-wise multiplication
            #
            #  Acumulate TM resulting maps for pyramid scales
            # ================================================
            if scale_fusion == 'sum':
                map_fusion_pyr = cv2.add(tm_map, map_fusion_pyr)
            else:
                print("Error: worng fusion map strategy.")
                return None

        # # Show maps for each template after fusion scales
        # plt.matshow(map_fusion_pyr, cmap='gray')
        # plt.show()

        # ================================================
        #               FUSION TEMPLATES
        #
        # @NOTE: Another approach could be used to fusion
        #        pyramid level results
        #           - sum
        #           - max value
        #           - pixel-wise multiplication
        #
        #  Acumulate TM resulting maps for template images
        # ================================================
        if tmpl_fusion == 'sum':
            map_fusion_tmpl = cv2.add(map_fusion_pyr, map_fusion_tmpl)
        else:
            print("Error: worng fusion map strategy.")
            return None

    # # Show maps for each image after fusion templates
    # plt.matshow(map_fusion_tmpl, cmap='gray')
    # plt.show()

    return map_fusion_tmpl


def build_pyramid(img=None, n_levels=3):
    """!@brief
    """
    # Assertions
    assert img is not None
    assert 0 < n_levels < 5

    # Build Gaussian pyramid resolution according 
    # to the number of levels required
    # -------------------------------------------
    out_pyramid = []

    # Levels: 1
    #   0: original
    if n_levels == 1:
        out_pyramid.append(img)

    # Levels: 2
    #   0: original     /\
    #   1: up_2        /__\
    elif n_levels == 2:
        # Levels
        lvl_0 = cv2.pyrUp(img, dstsize=(int(img.shape[1]*2), int(img.shape[0]*2)))
        lvl_1 = img

        # Output
        out_pyramid.append(lvl_0)
        out_pyramid.append(lvl_1)

    # Levels: 3
    #   2: down_2       /\
    #   1: original    /  \
    #   0: up_2       /____\
    elif n_levels == 3:
        # Levels
        lvl_0 = cv2.pyrUp(img, dstsize=(int(img.shape[1]*2), int(img.shape[0]*2)))
        lvl_1 = img
        lvl_2 = cv2.pyrDown(img, dstsize=(int(img.shape[1]/2), int(img.shape[0]/2)))

        # Output
        out_pyramid.append(lvl_0)
        out_pyramid.append(lvl_1)
        out_pyramid.append(lvl_2)

    # Levels: 4
    #    3: down_2       /\
    #    2: original    /  \
    #    1: up_2       /    \
    #    0: up_4      /______\
    elif n_levels == 4:
        # Levels
        dst = cv2.pyrUp(img, dstsize=(int(img.shape[1]*2), int(img.shape[0]*2)))
        lvl_0 = cv2.pyrUp(dst, dstsize=(int(dst.shape[1]*2), int(dst.shape[0]*2)))
        lvl_1 = cv2.pyrUp(img, dstsize=(int(img.shape[1]*2), int(img.shape[0]*2)))
        lvl_2 = img
        lvl_3 = cv2.pyrDown(img, dstsize=(int(img.shape[1]/2), int(img.shape[0]/2)))

        # Output
        out_pyramid.append(lvl_0)
        out_pyramid.append(lvl_1)
        out_pyramid.append(lvl_2)
        out_pyramid.append(lvl_3)

    # Default
    else:
        print("Number of pyramid levels much lower/bigger, try a number in the range [1,4].")
        return None

    # Resize images to odd sizes if needed
    # ------------------------------------
    for i in range(n_levels):
        rows = out_pyramid[i].shape[0]
        cols = out_pyramid[i].shape[1]
        flag = False

        if (rows % 2) == 0:
            rows = rows + 1
            flag = True

        if (cols % 2) == 0:
            cols = cols + 1
            flag = True

        if flag:
            out_pyramid[i] = cv2.resize(out_pyramid[i], dsize=(cols, rows))

    return out_pyramid