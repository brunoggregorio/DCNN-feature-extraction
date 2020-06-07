"""!
"""

import numpy as np
from dcnn_mtm import dcnn_mtm

# Create global variables
# =======================
base_path = "/home/brunoggregorio/Workspace/data/dataset/"
dcnn_model = ['Xception',
              'VGG16', 'VGG19',
              'ResNet50', 'ResNet101', 'ResNet152',
              'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
              'InceptionV3', 'InceptionResNetV2',
              'DenseNet121', 'DenseNet169', 'DenseNet201',
              'NASNetLarge']

min_side = 1000
max_side = 1400
thres_feature = 0.9
retained_value = 0.1
radius_feature = 5.0
pyramid_levels = 1
thres_ecc = 0.62
constant = -25.5
thres_binary = np.arange(0.7, 0.96, 0.05)


# =====================================================
#                   SPINALCORD 1
# =====================================================
video = "spinalcord_1"
aux_dcnn_model = ['ResNet152V2',
                  'InceptionV3', 'InceptionResNetV2',
                  'DenseNet121', 'DenseNet169', 'DenseNet201',
                  'NASNetLarge']
folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

# Getting started
print("------------------------------")
print("Processing video:", video)
print("------------------------------")

# For each model in the list
for model in aux_dcnn_model:

    # For each number of templates
    for n_tmpl in ['1', '2', '3']:
        # Get proper file of template positions
        template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

        print("Model: {}, # Templates: {}".format(model, n_tmpl))

        # For each threshold for TM image binarization
        for thresh in thres_binary:
            # Do the real work
            dcnn_mtm(folder_path=folder_path,
                     mask_img=mask_img,
                     ground_truth=ground_truth,
                     template=template,
                     dcnn_model=model,
                     thres_binary=thresh)


# =====================================================
#                    MESENTERY 1
# =====================================================
video = "mesentery_1"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

# Getting started
print("------------------------------")
print("Processing video:", video)
print("------------------------------")

# For each model in the list
for model in dcnn_model:

    # For each number of templates
    for n_tmpl in ['1', '2', '3']:
        # Get proper file of template positions
        template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

        print("Model: {}, # Templates: {}".format(model, n_tmpl))

        # For each threshold for TM image binarization
        for thresh in thres_binary:
            # Do the real work
            dcnn_mtm(folder_path=folder_path,
                     mask_img=mask_img,
                     ground_truth=ground_truth,
                     template=template,
                     dcnn_model=model,
                     thres_binary=thresh)


# =====================================================
#                    CREMASTER 2
# =====================================================
video = "cremaster_2"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

# Getting started
print("------------------------------")
print("Processing video:", video)
print("------------------------------")

# For each model in the list
for model in dcnn_model:

    # For each number of templates
    for n_tmpl in ['1', '2', '3']:
        # Get proper file of template positions
        template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

        print("Model: {}, # Templates: {}".format(model, n_tmpl))

        # For each threshold for TM image binarization
        for thresh in thres_binary:
            # Do the real work
            dcnn_mtm(folder_path=folder_path,
                     mask_img=mask_img,
                     ground_truth=ground_truth,
                     template=template,
                     dcnn_model=model,
                     thres_binary=thresh)


# =====================================================
#                    CREMASTER 1
# =====================================================
video = "cremaster_1"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

# Getting started
print("------------------------------")
print("Processing video:", video)
print("------------------------------")

# For each model in the list
for model in dcnn_model:

    # For each number of templates
    for n_tmpl in ['1', '2', '3']:
        # Get proper file of template positions
        template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

        print("Model: {}, # Templates: {}".format(model, n_tmpl))

        # For each threshold for TM image binarization
        for thresh in thres_binary:
            # Do the real work
            dcnn_mtm(folder_path=folder_path,
                     mask_img=mask_img,
                     ground_truth=ground_truth,
                     template=template,
                     dcnn_model=model,
                     thres_binary=thresh)


# =====================================================
#                     BRAIN 2
# =====================================================
video = "brain_2"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

# Getting started
print("------------------------------")
print("Processing video:", video)
print("------------------------------")

# For each model in the list
for model in dcnn_model:

    # For each number of templates
    for n_tmpl in ['1', '2', '3']:
        # Get proper file of template positions
        template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

        print("Model: {}, # Templates: {}".format(model, n_tmpl))

        # For each threshold for TM image binarization
        for thresh in thres_binary:
            # Do the real work
            dcnn_mtm(folder_path=folder_path,
                     mask_img=mask_img,
                     ground_truth=ground_truth,
                     template=template,
                     dcnn_model=model,
                     thres_binary=thresh)