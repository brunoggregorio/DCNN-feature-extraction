"""!
"""


import os
import cv2
import natsort
import numpy as np
import pandas as pd


def get_statistical_measures(gt=None,
                             det=None,
                             r=5):
    """!@brief
    """
    # Assertions
    assert gt is not None
    assert det is not None

    # Prepare data
    gt['count'] = 'FN'
    det['count'] = 'FP'

    # Loop for all frames
    # ===================
    for f in gt.frame.unique():

        # Compute measures for each frame individualy
        gt_frame = gt[gt['frame'] == f]

        # Find matchings and accumulate corresponding values
        for i_gt, r_gt in gt_frame.iterrows():
            # Get sub-set of detected points
            det_fp = det[(det['frame'] == f) & (det['count'] == 'FP')]
            if det_fp.empty:
                continue

            min_dist = r+1

            # Iterate in det array
            for i_det, r_det in det_fp.iterrows():
                dist = np.sqrt((r_gt['x']-r_det['x'])**2 + (r_gt['y']-r_det['y'])**2)

                if dist <= r and dist < min_dist:
                    min_dist = dist
                    min_idx = i_det

            # Found a matching point
            if min_dist <= r:
                gt['count'].at[i_gt] = 'TP'
                det['count'].at[min_idx] = 'TP'

    # Get countings
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

    return [TP, FP, FN, P, R, F1], gt, det


def show_output_images(folder=None,
                       gt=None,
                       det=None,
                       radius=8):
    """!@brief
    """
    # Assertions
    assert folder is not None
    assert gt is not None
    assert det is not None

    # Loop for all images inside the folder
    # =====================================
    for i, f in enumerate(natsort.natsorted(os.listdir(folder))):
        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):

            # Complete image path
            img_path = folder + f
            img = cv2.imread(img_path)

            # Filter and draw TP points
            # =========================
            tps = det[(det['count'] == 'TP') & (det['frame'] == i+1)]
            for _, r_tp in tps.iterrows():
                # Draw a circle with green line borders
                img = cv2.circle(img=img,
                                 center=(r_tp['x'], r_tp['y']),
                                 radius=radius,
                                 color=(0, 255, 0),
                                 thickness=1)

            tps = gt[(gt['count'] == 'TP') & (gt['frame'] == i+1)]
            for _, r_tp in tps.iterrows():
                # Draw a cross in the manual annotated centroid
                cv2.drawMarker(img=img,
                               position=(r_tp['x'], r_tp['y']),
                               color=(255, 0, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=4,
                               thickness=1)

            # Filter and draw FP points
            # =========================
            fps = det[(det['count'] == 'FP') & (det['frame'] == i+1)]
            for _, r_fp in fps.iterrows():
                # Draw a circle with blue line borders
                img = cv2.circle(img=img,
                                 center=(r_fp['x'], r_fp['y']),
                                 radius=radius,
                                 color=(255, 0, 0),
                                 thickness=1)

            # Filter and draw FN points
            # =========================
            fns = gt[(gt['count'] == 'FN') & (gt['frame'] == i+1)]
            for _, r_fn in fns.iterrows():
                top_left = (int(round(r_fn['x']-radius)),
                            int(round(r_fn['y']-radius)))
                bottom_right = (int(round(r_fn['x']+radius)),
                                int(round(r_fn['y']+radius)))
                # Draw a rectangle with red line borders
                img = cv2.rectangle(img=img,
                                    pt1=top_left,
                                    pt2=bottom_right,
                                    color=(0, 0, 255),
                                    thickness=1)

            # Displaying the image
            cv2.imshow(folder, img)
            cv2.waitKey(0)
            # cv2.imwrite('output_imgs/'+f, img)


def formatImagesForPCA(vec_imgs=None):
    """!@brief
    """
    assert vec_imgs is not None

    # # Using only Numpy
    # import numy as np
    #
    # M = None
    # for i in range(vec_imgs.shape[2]):
    #     img_as_row = vec_imgs[:,:,i].reshape(vec_imgs.shape[0] * vec_imgs.shape[1])
    #     try:
    #         M = np.vstack((M, img_as_row))
    #     except:
    #         M = img_as_row
    #
    # # Normalize the data [0,1]
    # M = cv2.normalize(M, 0, 1, norm_type=cv2.NORM_MINMAX)

    # Using pandas DataFrame for better manipulation
    M = pd.DataFrame([])
    for i in range(vec_imgs.shape[2]):
        img = pd.Series(vec_imgs[:, :, i].flatten(), name="Feature #"+str(i))
        img = (img - img.min()) / (img.max() - img.min()) # Normalization
        M = M.append(img)

    return M
