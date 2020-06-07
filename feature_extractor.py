"""!

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input


class FeatureExtractor:
    """!@brief

    @param model_name : The DCNN model name to be used to extract the features.
    @param base_model : The trainde DCNN model itself.
    @param model      : The intermediate layer model used to predict the input image.
    """
    base_model = None
    model_name = None
    model = None


    def __init__(self, model=None):
        """!@brief
        """
        self.model_name = model

        # Silence Tensorflow
        # ====================
        #   Levels:
        #       0: Everything
        #       1: Warnings
        #       2: Errors
        #       3: Fatal erros
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # ======================================
        # Load the pretrained models on ImageNet
        # ======================================

        # Xception
        # ========
        if self.model_name.lower() == "xception":
            layer = 'block4_sepconv1_act'
            
            self.base_model = tf.keras.applications.Xception(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # VGG16
        # =====
        elif self.model_name.lower() == "vgg16":
            layer = 'block3_conv1'
            
            self.base_model = tf.keras.applications.VGG16(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # VGG19
        # =====
        elif self.model_name.lower() == "vgg19":
            layer = 'block3_conv1'
            
            self.base_model = tf.keras.applications.VGG19(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # ResNet50
        # ========
        elif self.model_name.lower() == "resnet50":
            layer = 'conv2_block3_out'

            self.base_model = tf.keras.applications.ResNet50(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # ResNet101
        # =========
        elif self.model_name.lower() == "resnet101":
            layer = 'conv2_block3_out'

            self.base_model = tf.keras.applications.ResNet101(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # ResNet152
        # =========
        elif self.model_name.lower() == "resnet152":
            layer = 'conv2_block3_out'
            
            self.base_model = tf.keras.applications.ResNet152(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # ResNet50V2
        # ==========
        elif self.model_name.lower() == "resnet50v2":
            layer = 'conv2_block3_out'
            
            self.base_model = tf.keras.applications.ResNet50V2(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # ResNet101V2
        # ===========
        elif self.model_name.lower() == "resnet101v2":
            layer = 'conv2_block3_out'
            
            self.base_model = tf.keras.applications.ResNet101V2(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # ResNet152V2
        # ===========
        elif self.model_name.lower() == "resnet152v2":
            layer = 'conv2_block3_out'
            
            self.base_model = tf.keras.applications.ResNet152V2(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # InceptionV3
        # ===========
        elif self.model_name.lower() == "inceptionv3":
            # layer = 'activation_22'

            self.base_model = tf.keras.applications.InceptionV3(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            layer = self.base_model.layers[66].name

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # InceptionResNetV2
        # =================
        elif self.model_name.lower() == "inceptionresnetv2":
            layer = 'block35_1_ac'
            
            self.base_model = tf.keras.applications.InceptionResNetV2(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # DenseNet121
        # ===========
        elif self.model_name.lower() == "densenet121":
            layer = 'pool2_relu'
            
            self.base_model = tf.keras.applications.DenseNet121(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # DenseNet169
        # ===========
        elif self.model_name.lower() == "densenet169":
            layer = 'pool2_relu'
            
            self.base_model = tf.keras.applications.DenseNet169(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # DenseNet201
        # ===========
        elif self.model_name.lower() == "densenet201":
            layer = 'pool2_relu'
            
            self.base_model = tf.keras.applications.DenseNet201(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # NASNetLarge
        # ===========
        elif self.model_name.lower() == "nasnetlarge":
            # layer = 'activation'

            self.base_model = tf.keras.applications.NASNetLarge(
                                input_tensor=None,
                                weights='imagenet',
                                include_top=False)

            layer = self.base_model.layers[3].name

            self.model = tf.keras.Model(inputs=self.base_model.input,
                                        outputs=self.base_model.get_layer(layer).output)

        # Default
        # =======
        else:
            self.base_model = None
            self.model = None


    def _rescale_image(self, img, min_side=1000, max_side=1400):
        """!@brief
        Resize an image such that the size is constrained to min_side and max_side.

        @param min_side: The image's min side will be equal to min_side after resizing.
        @param max_side: If after resizing the image's max side is above max_side,
                         resize until the max side is equal to max_side.

        @return
            A resized image.
        """
        # Compute scale to resize the image
        # =================================
        (rows, cols, _) = img.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # Resize the image with the computed scale
        # ========================================
        img = cv2.resize(img, None, fx=scale, fy=scale)

        return img


    def get_output(self,
                   img=None,
                   min_side=1000,
                   max_side=1400,
                   resize=True):
        """!@brief
        """
        # Assertions
        assert img is not None

        # Preprocess input image
        original_size = (img.shape[1], img.shape[0])

        # Resize image accordingly to the model
        # =====================================
        if self.model_name.lower() == "nasnetlarge":
            x = cv2.resize(img, (331, 331), cv2.INTER_LINEAR)
        else:
            x = self._rescale_image(img, min_side, max_side)

        # Adjust input image shape
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)

        # @todo Check if preprocessing is present in all models
        # x = preprocess_input(x)

        # Predict the input image in the intermediate layer
        output_model = self.model.predict(x)[0, :, :]

        # Return images to their original size
        if resize:
            output_resized = np.zeros(shape=(img.shape[0], img.shape[1], output_model.shape[2]))
            for i in range(output_model.shape[2]):
                output_resized[:, :, i] = cv2.resize(output_model[:, :, i],
                                                     original_size,
                                                     cv2.INTER_LINEAR)
        else:
            output_resized = output_model

        # # Debug sizes and model output
        # print("Original size (high|rows|shape[0], width|cols|shape[1], n_channels):", img.shape)
        # print("Model input image size:", x.shape)
        # print("Number of output images in the intermediate layer:", output_model.shape[2])
        # print("Output image size:", output_resized[:, :, 0].shape)

        return output_resized


    def clean_model(self):
        """!@breif
        """
        tf.keras.backend.clear_session()
        del self.model