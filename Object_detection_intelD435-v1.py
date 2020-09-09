'''
This code ONLY works when Intel Realsense D435 is connected with USB3.0 using its own/original type C Realsense cable

Code by: Kürşat Coşkun [28 May 2019]
https://github.com/kursatcoskun

Modified by: Mohamad Zarif bin Ramzan [7 Aug 2020]
'''

import pyrealsense2 as rs
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
import datetime

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 5

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# This is the minimal recommended resolution for D435
config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
# Getting the depth sensor's depth scale
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("\n\t depth_scale: " + str(depth_scale))

# Getting the depth sensor's depth scale
pipeline.stop()

# Start streaming
pipeline.start(config)
try:
    while True:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorImage=np.array(color_frame.get_data())
        #depthImage=np.array(depth_frame.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        frame_expanded = np.expand_dims(colorImage, axis=0)
        depth_expanded = np.expand_dims(depth_colormap, axis=0)
        
        # Perform the actual detection by running the model with the image as input - Depth
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: depth_expanded})

        # Perform the actual detection by running the model with the image as input - RGB
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
            
        # Draw the results of the detection (aka 'visulaize the results') - Depth 
        vis_util.visualize_boxes_and_labels_on_image_array(
            depth_colormap,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.60)
        
        # Draw the results of the detection (aka 'visulaize the results') - RGB
        vis_util.visualize_boxes_and_labels_on_image_array(
            colorImage,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.60)

        # Stack both images horizontally
        images = np.hstack((colorImage, depth_colormap))
        cv2.imshow('Fine Grained RGB-D Object Recognition using Realsense D435', images)
        
        #cv2.namedWindow('Fine Grained RGB-D Object Recognition using Realsense D435', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Realsense', colorImage)
        #cv2.imshow('Depth Cam',depth_colormap)
        cv2.waitKey(1)
finally:

    # Stop streaming
    pipeline.stop()


