import sys
sys.path.append('./nested-transformer')

import os
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
from configs import imagenet_nest  

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")
logging.set_verbosity(logging.INFO)

print("JAX devices:\n" + "\n".join([repr(d) for d in jax.devices()]))
print('Current folder content', os.listdir())
