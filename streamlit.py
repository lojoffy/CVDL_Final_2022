import streamlit as st

import sys
sys.path.append('./nested-transformer')

import os
import cv2
import time
import flax
from flax import nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
from absl import logging

from libml import input_pipeline 
from libml import preprocess
from models import nest_net  
import train  
from configs import cifar_nest 
from configs import cifar_nest_large
from configs import imagenet_nest  

from keras.utils.vis_utils import plot_model

import PIL


dataset_prefix = "/Users/blueshiner/git/3_Collections/32_CV2/"
#dataset_prefix = "/app/ncku_cv2022_hw1/"
folder_1 = dataset_prefix + "dataset/Dataset_CvDl_Hw2/Q1_Image/"

# ------- init setting -------
st.set_page_config(
    page_title="NCKU CVDL Final",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title('NCKU - Computer Vision and Deep Learing 2022')
st.subheader("Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding")
# ------- end -------
st.caption("Author: F14071075@2022")

# Stable Variances:
if 'state_1' not in st.session_state:
    st.session_state['state_1'] = 0
if 'state_2' not in st.session_state:
    st.session_state['state_2'] = 0
if 'state_3' not in st.session_state:
    st.session_state['state_3'] = 0
if 'state_4' not in st.session_state:
    st.session_state['state_4'] = 0
if 'state_5' not in st.session_state:
    st.session_state['state_5'] = 0

# used function
def no_fn():
    pass


# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

st.info("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))

if st.button("Load Checkpoints:"):
    st.session_state.state_1 = 1
if st.session_state.state_1 >= 1:
    st.warning("Please select a pretrained model")
    opt_model = st.selectbox(
        'Select a pretrained model.',
        ('nest_tiny_s16_32', 'nest_small_s16_32'))

    if opt_model == "nest_tiny_s16_32":
        checkpoint_dir = "./checkpoints/nest_cifar/checkpoints-0/"
        cifar_nest_config = cifar_nest.get_config()

    if opt_model == "nest_small_s16_32":
        checkpoint_dir = "./checkpoints/nest_cifar_large/checkpoints-0/"
        cifar_nest_config = cifar_nest_large.get_config()


    #imagenet_config = imagenet_nest.get_config()

    state_dict = train.checkpoint.load_state_dict(
        checkpoint_dir)
        #os.path.join(checkpoint_dir, os.path.basename(checkpoint_dir_exp)))
    variables = {
        "params": state_dict["optimizer"]["target"],
    }
    variables.update(state_dict["model_state"])
    #model_cls = nest_net.create_model(imagenet_config.model_name, imagenet_config)
    model_cls = nest_net.create_model(cifar_nest_config.model_name, cifar_nest_config)
    #model = functools.partial(model_cls, num_classes=1000)
    model = functools.partial(model_cls, num_classes=10)

if st.button("Load Required Data:"):
    st.session_state.state_1 = 2
if st.session_state.state_1 >= 2:
    st.info("Loaded Succesfully.")
    with st.expander("Show Dataset Info"):
        st.write("Dataset: CIFAR-10")
        st.image("Figs/Fig1_CIFAR-10.png")
    with st.expander("Show Model Parameters"):
        #st.code("Using augmentation randaugment with parameters {'cutout': False, 'magnitude': 9, 'magstd': 0.5, 'num_layers': 2, 'prob_to_apply': 0.5, 'size': 32}")
        st.code(cifar_nest_config)
    with st.expander("Show Training logging"):
        st.write("Evaluation accuracy")
        st.image("Figs/1.png")
        st.write("Evaluation loss")
        st.image("Figs/2.png")
        st.write("L2 gradients")
        st.image("Figs/3.png")
        st.write("Learning rate")
        st.image("Figs/4.png")
        st.write("Training loss")
        st.image("Figs/5.png")
        st.write("Training loss std")
        st.image("Figs/6.png")
        st.write("Training accuracy")
        st.image("Figs/7.png")
    with st.expander("Show Model Structure:"):
        pass


    image_0 = st.file_uploader("Upload Image", type=['jpg', 'png', 'bmp'])
    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)
        cv2.imwrite('process.jpg', cv_image_0)
        st.image(cv_image_0, width=256)


        def predict(image):
            logits = model(train=False).apply(variables, image, mutable=False)
            # Return predicted class and confidence.
            return logits.argmax(axis=-1), nn.softmax(logits, axis=-1).max(axis=-1)

        def _preprocess(image):
            #image = np.array(image.resize((224, 224))).astype(np.float32) / 255
            image = np.array(image.resize((32, 32))).astype(np.float32) / 255
            mean = np.array(preprocess.IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
            std = np.array(preprocess.IMAGENET_DEFAULT_STD).reshape(1, 1, 3)
            image = (image - mean) / std
            return image[np.newaxis,...]

        def _get_class(index):
            class_lst = ["airplane",
                    "automobile",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                    ]
            return class_lst[index]

        img = PIL.Image.open('process.jpg')
        input = _preprocess(img)

        cls, prob = predict(input)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"CIFAR-10 class id: ")
            st.success(f"{cls[0]}")
        with col2:
            st.info(f"class name: ")
            st.success(f"{_get_class(cls[0])}")
        with col3:
            st.info(f"prob: ")
            st.success(f"{prob[0]}")
