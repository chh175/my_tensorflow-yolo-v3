import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny
import matplotlib.pyplot as plt 

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image



input_img = 'image2.jpg'
img = Image.open(input_img)
#img_resized = letter_box_image(img, 416, 416, 128)

img_resized = np.array(img.resize((416, 416), resample=Image.BILINEAR))
img_resized = img_resized.astype(np.float32)

classes = load_coco_names('coco.names')

frozenGraph = load_graph('frozen_darknet_yolov3_model.pb')

boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)


with tf.Session(graph=frozenGraph) as sess:
    
    detected_boxes = sess.run(
        boxes, feed_dict={inputs: [img_resized]})


filtered_boxes = non_max_suppression(detected_boxes,
                                     confidence_threshold=0.5,
                                     iou_threshold=0.4)

draw_boxes(filtered_boxes, img, classes, (416, 416), False)




#import pylab
#pylab.rcParams['figure.figsize'] = (15.0, 8.0) # 显示大小
#plt.imshow(img)

img.save('o1.jpg')
