"""!
"""


import os
import cv2
import natsort
import argparse
import numpy as np
import pandas as pd
from feature_extractor import FeatureExtractor
from mtm import get_best_features, template_matching, build_pyramid
from postprocessing import postprocessing
from utils import get_statistical_measures, show_output_images


def get_args():
    """!@brief
    -------------------------------------------------------------------------
    | Video | Avg_size | Radius |        Template       | BS_1 | BS_2 | BS_3 |
    |-------|----------|--------|-----------------------|------|------|------|
    | B1    | 7        | 3.5    | [312, 159, 323, 172]  | 13   | 13   | 15   |
    | B2    | 7        | 3.5    | [113, 178, 122, 187]  | 9    | 9    | 19   |
    | C1    | 25       | 12.5   | [251, 120, 308, 181]  | 61   | 59   | 63   |
    | C2    | 5        | 2.5    | [994, 578, 1009, 595] | 17   | 17   | 15   |
    | SC    | 5        | 2.5    | [219, 178, 230, 193]  | 15   | 15   | 15   |
    | ME    | 11       | 5.5    | [245, 216, 268, 237]  | 23   | 29   | 25   |
    -------------------------------------------------------------------------

    Labels:
        - Avg_size: The average size of all cells in the video.
        - Radius: Avg_size / 2.
        - Template: The main (first) template coordinates.
        - BS_x: The block size value for the number x of templates used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", help="Increase output verbosity.", action="store_false")
    parser.add_argument("-f", "--folder-path", type=str, help="Folder path for the input images.")
    parser.add_argument("-m", "--mask-img", type=str, help="Mask image file.")
    parser.add_argument("-g", "--ground-truth", type=str, help="Ground truth text file.")
    parser.add_argument("-t", "--template", type=str, help="Text file with templates positions.")
    parser.add_argument("-d", "--dcnn-model", type=str, help="DCNN model to extract features.", default="resnet101")
    parser.add_argument("-o", "--output", type=str, help="Output verbosity file.", default=None)
    parser.add_argument("-p", "--output_points", type=str, help="Output detections file.", default=None)

    parser.add_argument("-mins", "--min-side", type=int, help="Image min side.", default=1000)
    parser.add_argument("-maxs", "--max-side", type=int, help="Image max side.", default=1400)
    parser.add_argument("-tfea", "--thres-feature", type=float, help="TM threshold for features selection.", default=0.9)
    parser.add_argument("-ret", "--retained-value", type=float, help="Retained value for features selection.", default=0.1)
    parser.add_argument("-rfea", "--radius-feature", type=float, help="Radius value for features selection.", default=5.0)
    parser.add_argument("-tbin", "--thres-binary", type=float, help="Threshold value for binarization.", default=0.9)
    parser.add_argument("-npyr", "--pyramid-levels", type=int, help="Number of template pyramid levels.", default=1)
    parser.add_argument("-tecc", "--thres-ecc", type=float, help="Threshold value for eccentricity post-processing.", default=0.62)
    parser.add_argument("-cons", "--constant", type=float, help="Constant value for Adaptive Thresholding.", default=-25.5)

    return parser.parse_args()


def dcnn_mtm(verbosity=False,
             folder_path=None,
             mask_img=None,
             ground_truth=None,
             template=None,
             dcnn_model=None,
             output=None,
             output_points=None,
             min_side=1000,
             max_side=1400,
             thres_feature=0.9,
             retained_value=0.1,
             radius_feature=5.0,
             thres_binary=0.9,
             pyramid_levels=1,
             thres_ecc=0.62,
             constant=-25.5):
    """!@brief
    """
    # Read params
    mask_img = cv2.imread(mask_img, 0)
    gt = pd.read_csv(ground_truth, sep='\t', header=None, names=['frame', 'x', 'y'])

    # Extract image features from a DCNN model
    extractor = FeatureExtractor(dcnn_model)

    # ================================================
    #               FOR THE FIRST FRAME
    #
    #   1. Get output model feature images
    #   2. Find the best set of feature images
    #   3. Extract templates from ROIs pre-defined
    #   4. Compute and save all templates accordingly
    # ================================================

    # Complete image path
    img_path = folder_path + natsort.natsorted(os.listdir(folder_path))[0]
    first_frame = cv2.imread(img_path)

    # Select ROI for template images
    if template is None:
        # Manual templates selection
        templates = cv2.selectROIs("First frame", first_frame)
    else:
        # Previous selected templates
        templates = np.loadtxt(template, delimiter=' ').astype(int)

    # Adjust templates shape
    if len(templates.shape) == 1:
        templates = [templates]

    # Get the block size value from template sizes
    block_size = 0.0
    for t in templates:
        distX = np.abs(t[0]-t[2])
        distY = np.abs(t[1]-t[3])
        block_size = np.max([block_size, distX, distY])

    # Get image features and format them
    ext_outputs = extractor.get_output(img=first_frame,
                                       min_side=min_side,
                                       max_side=max_side,
                                       resize=True)

    # Find best features from model output
    # ====================================
    # n_ext_outputs = ext_outputs.shape[2]
    # print("Number of model features:", n_ext_outputs)

    feature_idxs = get_best_features(features=ext_outputs,
                                     threshold=thres_feature,
                                     retained_value=retained_value,
                                     radius=radius_feature,
                                     template=templates[0],
                                     mask_img=mask_img,
                                     gt=gt.copy())

    # print("Number of selected features:", len(feature_idxs))
    # print("Feature image indexes:", feature_idxs)

    # ======================================================================
    # # DEBUG
    # for feature_img in np.rollaxis(ext_outputs[:, :, feature_idxs], axis=2):
    #     import matplotlib.pyplot as plt
    #     plt.matshow(feature_img, cmap='gray')
    #     plt.show()
    # ======================================================================


    # Get template images accordingly
    # @NOTE
    #   templates_pool[feat][templ][scale]
    # ====================================
    templates_pool = []

    # Loop for each feature of first frame
    for feature_img in np.rollaxis(ext_outputs[:, :, feature_idxs], axis=2):
        tmpl_feature = []

        # Normalize template image
        feature_img = cv2.normalize(src=feature_img,
                                    dst=None,
                                    alpha=0,
                                    beta=255,
                                    norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)

        # Loop for each template image
        for tmpl_pos in templates:
            tmpl = feature_img[tmpl_pos[1]:tmpl_pos[3],
                               tmpl_pos[0]:tmpl_pos[2]] # [y0:y1, x0:x1]

            # Get template scales (pyramid)
            tmpl_scale = build_pyramid(tmpl, pyramid_levels)

            tmpl_feature.append(tmpl_scale)
        templates_pool.append(tmpl_feature)

    # Do the same previous task to the original image channels
    # ========================================================

    # When image has only one channel -> reshape
    if len(first_frame.shape) == 2:
        img_aux = first_frame.reshape((first_frame.shape[0], first_frame.shape[1], 1))
    else:
        img_aux = first_frame.copy()

    # Loop in each channel
    for channel_img in np.rollaxis(img_aux, axis=2):
        tmpl_feature = []

        # Normalize template image
        channel_img = cv2.normalize(src=channel_img,
                                    dst=None,
                                    alpha=0,
                                    beta=255,
                                    norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)

        # Loop for each template image
        for tmpl_pos in templates:
            tmpl = channel_img[tmpl_pos[1]:tmpl_pos[3],
                               tmpl_pos[0]:tmpl_pos[2]] # [y0:y1, x0:x1]

            # Get template scales (pyramid)
            tmpl_scale = build_pyramid(tmpl, pyramid_levels)

            tmpl_feature.append(tmpl_scale)
        templates_pool.append(tmpl_feature)

    # ==============================================================
    # # DEBUG
    # for ff in range(len(templates_pool)):
    #     for tt in range(len(templates)):
    #         for ss in range(pyramid_levels):
    #             # Show all templates
    #             import matplotlib.pyplot as plt
    #             plt.matshow(templates_pool[ff][tt][ss], cmap='gray')
    #             plt.show()
    # ==============================================================

    # ========================================
    # Read all images in the folder
    # @TODO
    #      Read frames directly from a video.
    # ========================================

    # Open a text file to save output values
    if output is not None:
        F = open(output, 'a')

    all_det = pd.DataFrame() # all detection points
    # rows = ext_outputs[:, :, 0].shape[0]
    # cols = ext_outputs[:, :, 0].shape[1]
    # fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    # out = cv2.VideoWriter('output_map.avi', fourcc, 16.0, (rows, cols))

    for i, f in enumerate(natsort.natsorted(os.listdir(folder_path))):
        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):

            if verbosity:
                # print("------------------------------------")
                print("Processing frame", f)
                # print("------------------------------------")

            # Complete image path
            img_path = folder_path + f
            img = cv2.imread(img_path)

            # Get image features and format them
            feature_imgs = extractor.get_output(img)[:, :, feature_idxs]

            # When image has only one channel -> reshape
            if len(img.shape) == 2:
                img_aux = img.reshape((img.shape[0], img.shape[1], 1))
            else:
                img_aux = img.copy()

            # Append original frame to the feature images
            feature_imgs = np.append(feature_imgs, img_aux, axis=2)

            # Initialize fusion map for all feature images after TM
            output_tm_map = np.zeros_like(feature_imgs[:, :, 0], dtype=np.float32)

            # Loop for each feature image
            # ===========================
            for j in range(feature_imgs.shape[2]):

                # Apply Multiple Template Matching
                # ================================
                out_map = template_matching(image=feature_imgs[:, :, j],
                                            templates=templates_pool[j],
                                            scale_fusion='sum',
                                            tmpl_fusion='sum')

                # ================================================
                #              FUSION FEATURE MAPS
                #
                # @NOTE: Another approach could be used to fusion
                #        pyramid level results
                #           - sum
                #           - max value
                #           - pixel-wise multiplication
                #
                #  Acumulate TM resulting maps for feature images
                # ================================================
                output_tm_map = cv2.add(out_map, output_tm_map)

            # # Save output video
            # # =================
            # # Normalize output map
            # out_img = cv2.normalize(src=output_tm_map,
            #                         dst=None,
            #                         alpha=0,
            #                         beta=255,
            #                         norm_type=cv2.NORM_MINMAX,
            #                         dtype=cv2.CV_8U)
            # out_img_rgb = cv2.cvtColor(src=out_img,
            #                            code=cv2.COLOR_GRAY2BGR)
            # out.write(out_img_rgb)

            # # Show resulting image after fusion of feature maps
            # import matplotlib.pyplot as plt
            # plt.matshow(output_tm_map, cmap='gray')
            # plt.show()

            # ======================================
            #           POST-PROCESSING
            #   Get eccentricity of points in the
            #   template matching resulting maps.
            # ======================================
            det_frame = postprocessing(img=output_tm_map,
                                       mask=mask_img,
                                       threshold=thres_binary)

            if len(det_frame) > 0:
                for det in det_frame:
                    det['frame'] = i+1

                all_det = all_det.append(det_frame)

        # if i == 2:
        #     break

    # # Release video writer
    # out.release()

    # ===================================================
    # @IMPORTANT
    #       As the dataframe "all_det" is appended and
    #       has its item values changed by index inside
    #       postprocessing function, we need to reset it.
    # ===================================================
    all_det = all_det.reset_index(drop=True)

    # Get statistical measures
    #   meas = [TP, FP, FN, P, R, F1]
    # ================================
    meas, out_gt, out_det = get_statistical_measures(gt=gt.copy(),
                                                     det=all_det.copy(),
                                                     r=block_size/2)

    # Print on terminal
    print("TP = {}, FP = {}, FN = {}, P = {:.2f}, R = {:.4f}, " \
        "F1 = {:.4f} | THRESH: {:.2f}".format(meas[0], meas[1], meas[2],
                                              meas[3], meas[4], meas[5], thres_binary))

    if output is not None:
        # Print on file
        F.write("TP = {}, FP = {}, FN = {}, P = {:.2f}, R = {:.4f}, " \
                "F1 = {:.4f} | THRESH: {:.2f}\n".format(meas[0], meas[1], meas[2],
                                                        meas[3], meas[4], meas[5], thres_binary))

    # ==================================
    # Show all frames with corresponding
    # annotations and detections
    # ==================================
    if verbosity:
        show_output_images(folder=folder_path,
                           gt=out_gt,
                           det=out_det)

    # Save output detection points in text file
    if output_points:
        out_det[['frame', 'x', 'y']].to_csv(output_points,
                                            index=False,
                                            header=False,
                                            sep=' ')

    if output is not None:
        F.close()

    # Clear memory and graphs from keras
    if extractor is not None:
        extractor.clean_model()


if __name__ == "__main__":
    args = get_args()
    dcnn_mtm(verbosity=args.verbosity,
             folder_path=args.folder_path,
             mask_img=args.mask_img,
             ground_truth=args.ground_truth,
             template=args.template,
             dcnn_model=args.dcnn_model,
             output=args.output,
             output_points=args.output_points,
             min_side=args.min_side,
             max_side=args.max_side,
             thres_feature=args.thres_feature,
             retained_value=args.retained_value,
             radius_feature=args.radius_feature,
             thres_binary=args.thres_binary,
             pyramid_levels=args.pyramid_levels,
             thres_ecc=args.thres_ecc,
             constant=args.constant)
