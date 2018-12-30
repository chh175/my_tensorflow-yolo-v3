import numpy as np
import tensorflow as tf

from yolo_v3 import *

inputs = tf.placeholder(tf.float32, [None,320,320,3],'input')


detections = yolo_v3(inputs, 20)