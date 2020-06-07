"""!
"""

import numpy as np
from dcnn_mtm import dcnn_mtm

# Create global variables
# =======================
base_path = "/home/brunoggregorio/Workspace/data/dataset/"
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
#                     BRAIN 1
# =====================================================
video = "brain_1"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

n_tmpl = '3'
model = 'VGG19'
thresh = 0.85
template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

print("Video: {}, model={}, #tmpl={}, thresh={:.2f}".format(video, model, n_tmpl, thresh))

dcnn_mtm(folder_path=folder_path,
         mask_img=mask_img,
         ground_truth=ground_truth,
         template=template,
         dcnn_model=model,
         thres_binary=thresh,
         output_points='B1_output2D_centroids.txt',
         verbosity=False)


# =====================================================
#                     BRAIN 2
# =====================================================
video = "brain_2"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

n_tmpl = '3'
model = 'DenseNet121'
thresh = 0.90
template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

print("Video: {}, model={}, #tmpl={}, thresh={:.2f}".format(video, model, n_tmpl, thresh))

dcnn_mtm(folder_path=folder_path,
         mask_img=mask_img,
         ground_truth=ground_truth,
         template=template,
         dcnn_model=model,
         thres_binary=thresh,
         output_points='B2_output2D_centroids.txt',
         verbosity=False)


# =====================================================
#                   SPINALCORD 1
# =====================================================
video = "spinalcord_1"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

n_tmpl = '1'
model = 'ResNet50'
thresh = 0.80
template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

print("Video: {}, model={}, #tmpl={}, thresh={:.2f}".format(video, model, n_tmpl, thresh))

dcnn_mtm(folder_path=folder_path,
         mask_img=mask_img,
         ground_truth=ground_truth,
         template=template,
         dcnn_model=model,
         thres_binary=thresh,
         output_points='SC_output2D_centroids.txt',
         verbosity=False)


# =====================================================
#                   CREMASTER 1
# =====================================================
video = "cremaster_1"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

n_tmpl = '2'
model = 'VGG19'
thresh = 0.85
template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

print("Video: {}, model={}, #tmpl={}, thresh={:.2f}".format(video, model, n_tmpl, thresh))

dcnn_mtm(folder_path=folder_path,
         mask_img=mask_img,
         ground_truth=ground_truth,
         template=template,
         dcnn_model=model,
         thres_binary=thresh,
         output_points='C1_output2D_centroids.txt',
         verbosity=False)


# =====================================================
#                   CREMASTER 2
# =====================================================
video = "cremaster_2"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

n_tmpl = '3'
model = 'VGG16'
thresh = 0.80
template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

print("Video: {}, model={}, #tmpl={}, thresh={:.2f}".format(video, model, n_tmpl, thresh))

dcnn_mtm(folder_path=folder_path,
         mask_img=mask_img,
         ground_truth=ground_truth,
         template=template,
         dcnn_model=model,
         thres_binary=thresh,
         output_points='C2_output2D_centroids.txt',
         verbosity=False)


# =====================================================
#                   MESENTERY 1
# =====================================================
video = "mesentery_1"

folder_path = base_path + video + "/frames/"
mask_img = base_path + video + "/" + video + "_mask.png"
ground_truth = base_path + video + "/" + video + "_gt.txt"

n_tmpl = '2'
model = 'VGG16'
thresh = 0.75
template = base_path + video + "/" + video + "_" + n_tmpl + "-templates.txt"

print("Video: {}, model={}, #tmpl={}, thresh={:.2f}".format(video, model, n_tmpl, thresh))

dcnn_mtm(folder_path=folder_path,
         mask_img=mask_img,
         ground_truth=ground_truth,
         template=template,
         dcnn_model=model,
         thres_binary=thresh,
         output_points='ME_output2D_centroids.txt',
         verbosity=False)