
# coding: utf-8

# In[1]:


# Created: 5-Jul-2018
# This is a code version of the original MR but will receive detection
# information so that we can separate the processing whether it is 
# OpenCV-based (i.e. SIFT, ORB, classical, etc.) or TF-based.


# In[2]:


# coding: utf-8

# In[1]:


#!/usr/bin/env python

import cv2
import numpy as np
import sys
import time
import requests
import math

import os
import six.moves.urllib as urllib
import tensorflow as tf
import zipfile
import collections
import warnings
warnings.filterwarnings("ignore")

from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
from io import StringIO, BytesIO
from matplotlib import pyplot as plt
from PIL import Image


# In[3]:


# This is needed to add the tf-object detecton api modules
sys.path.append("/srv/jaybie/models/research")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# In[4]:


# Global parameters
detection_mode = "SIFT" #default to SIFT

# OpenCV parameters
search_params = dict(checks = 20) # this is for the flann-based matcher
largest = {4032, 3024}

ref = "res/unsw.png"
ref_img = cv2.imread(ref, 0)

# SIFT parameters
siftDetector = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
siftMatcher = cv2.FlannBasedMatcher(index_params, search_params)
rsKP, rsDES = siftDetector.detectAndCompute(ref_img, None)

# ORB parameters
orbDetector = cv2.ORB_create()
#descriptor = cv2.xfeatures2d.SIFT_create()
orbMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # for ORB keypoints
roKP, roDES = orbDetector.detectAndCompute(ref_img,None)


# In[5]:


# Main tasks

# What model to use.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
OBJ_API_BASE = '/srv/jaybie/models/research/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# In[6]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(OBJ_API_BASE + PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(OBJ_API_BASE + PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[7]:


def get_object_locations(
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """This is an adaptation of the function from the visualization_utils
  named 'visualize_boxes_and_labels_on_image_array' but does not embed the
  masks to the image directly but just extract the relative position of
  objects within the view. This is to prevent sharing raw visual informa-
  tion to intended third parties interested to know the detected objects.
  
  Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
  return box_to_display_str_map


# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[9]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[10]:


def tf_process_img(payload):

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    t0 = time.clock()
    
    image = Image.open(BytesIO(payload))
    image_np = load_image_into_numpy_array(image)
    
    #image_np = np.array(payload)
    #image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # Actual detection.
    t0_5 = time.clock()
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    results = get_object_locations(
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        min_score_thresh = .6,
        use_normalized_coordinates=True,
        line_thickness=8)

    t1 = time.clock()
    
    tf_detect = t0_5 - t0
    tf_io = t1-t0
    #print("Time to TF detect:", t0_5-t0, ", overall time to TF detect", t1-t0)

    return results, tf_detect, tf_io


# In[11]:


# In[2]:

def cv_process_img(payload):
    
    nparr = np.fromstring(payload, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
    result = []    
    MIN_MATCH_COUNT = math.ceil(30/(max(largest)/max(image.shape)))*10 # very relaxed matching at 10 matches minimum
    #print("MIN_MATCH_COUNT:", MIN_MATCH_COUNT)
    #print("Image size:",image.shape)
    query_img = image
    
    t0 = time.clock()

    if "ORB" in detection_mode:
        
        rKP, rDES = roKP, roDES
        qKP, qDES = orbDetector.detectAndCompute(query_img, None)
        
        try:
            good = orbMatcher.match(rDES,qDESA)
        except:
            state = "Matching Error: not enough query points."
            result.append(state)
            result.append(query_img)
            result.append(np.array([]))
            t1 = time.clock()
            cv_time = t1-t0 #print(state,"Time to process:", t1-t0)
            result.append((t1-t0))
            return result 
        
    else:
        # Default to SIFT if non-TF
        
        rKP, rDES = rsKP, rsDES
        qKP, qDES = siftDetector.detectAndCompute(query_img, None)
        
        try:
            matches = siftMatcher.knnMatch(rDES,qDES,k=2)
        except:
            state = "Matching Error: not enough query points."
            result.append(state)
            result.append(query_img)
            result.append(np.array([]))
            t1 = time.clock()
            cv_time = t1-t0 #print(state,"Time to process:", t1-t0)
            result.append((t1-t0))
            return result    

        # store all the good matches as per Lowe's ratio test.
        good = []
        distances = []

        for m,n in matches:
            distances.append(m.distance)
            if m.distance < 0.75*n.distance:
                good.append(m)
                
    good = sorted(good, key = lambda x:x.distance)

    if len(good)>MIN_MATCH_COUNT:
        state = "Enough matches: object is propbably in view."
        # extract location of points in both images
        src_pts = np.float32([ rKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ qKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # find the perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        # get the transform points in the (captured) query image
        h,w = ref_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        try:
            dst = cv2.perspectiveTransform(pts,M)
            # draw the transformed image
            result.append(state)
            result.append(cv2.drawContours(query_img,[np.int32(dst)],-1,(255,0,0),6))
            result.append(dst)
        except:
            state = "Error getting perspective transform."
            result.append(state)
            result.append(query_img)
            result.append(np.array([]))
        finally:
            t1 = time.clock()
            cv_time = t1-t0 #print(state,"Time to CV process:", t1-t0)
            result.append((t1-t0))

    else:
        state = "Not enough matches:"+ str(len(good))
        result.append(state)
        result.append(query_img)
        result.append(np.array([]))
        t1 = time.clock()
        cv_time = t1-t0 #print(state,"Time to CV process:", t1-t0)
        result.append((t1-t0))

    return result, cv_time


# In[12]:


def resultsToTree(tf_result, cv_result):
    mr_objects = Element("mr_objects")

    if tf_result:
        # Adding the results from the TF operation to the XmlTree
        #print(len(tf_result),"TF object/s")
        for box, name in tf_result.items():
            ymin, xmin, ymax, xmax = box
            object = SubElement(mr_objects,"object")
            SubElement(object, "type").text = "TF"
            SubElement(object, "name").text = str(name)
            SubElement(object, "ymin").text = str(ymin)
            SubElement(object, "xmin").text = str(xmin)
            SubElement(object, "ymax").text = str(ymax)
            SubElement(object, "xmax").text = str(xmax)
  
    if cv_result:
        cv_result_list = []
        for point in cv_result:
            cv_result_list.append(point[0])
        cv_exes = sorted(cv_result_list, key=lambda x: x[0])
        cv_eyes = sorted(cv_result_list, key=lambda x: x[1])
        # Adding the cv_results from the CV detection to the XmlTree
        object = SubElement(mr_objects,"object")
        SubElement(object, "type").text = "CV"
        SubElement(object, "name").text = "cv"
        SubElement(object, "ymin").text = str(cv_result_list[0][1]/300)
        SubElement(object, "xmin").text = str(cv_result_list[0][0]/300)
        SubElement(object, "ymax").text = str(cv_result_list[3][1]/300)
        SubElement(object, "xmax").text = str(cv_result_list[3][0]/300)

    tree = ElementTree(mr_objects)
    
    return tree


# In[13]:


# In[3]:

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

  # GET
    def do_GET(self):
        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()

        # Send message back to client
        message = "Hello world!"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return
    
    
    def do_POST(self):
        #print( "incoming http: ", self.path )
        t0 = time.clock()
        tf_detect = 0
        tf_io = 0
        cv_time = 0

        # Gets the parameters of the data
        detection_mode = self.headers['Detection-mode']
        content_length = int(self.headers['Content-Length'])
        content_type = self.headers['Content-type']
                
        if "image" in content_type:
            
            # Gets the data itself. Also, we over-catch by 16 bytes.
            post_data = self.rfile.read(content_length+16)

            #print("Length of content:", len(post_data))
            #print("Before pruning\n", post_data)

            # We remove the first set of bytes up until the first
            # carraige return and newline.
            for b in np.arange(len(post_data)):
               # Checking where the first newline is
                if post_data[b] == 13:
                    post_data = post_data[b+2:]
                    break
                else:
                    continue

            #stream = BytesIO(post_data)
            #tmp = 'tmp.jpg'
            #with open('tmp.jpg','wb') as out:
            #    out.write(post_data)

            #out.close()
            # Converting the byte buffer to an numpy array for opencv and tf
            #nparr = np.fromstring(post_data, np.uint8)
            #img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            try:
                if "TF" in detection_mode:
                    cv_result = []
                    cv_time = 0
                    tf_result, tf_detect, tf_io = tf_process_img(post_data)#post_data)#
                else:
                    cv_result, cv_time = cv_process_img(post_data)
                    tf_result = []
                    tf_detect = 0
                    tf_io = 0
                #cv_result, cv_time = cv_process_img(post_data)
                #tf_result, tf_detect, tf_io = tf_process_img(post_data)#post_data)#

                t1 = time.clock()
                resultTree = resultsToTree(tf_result,cv_result[2].tolist())
                resultTree.write("result.xml")

                # Send message back to client
                # Write content as utf-8 data
                #print(len(final_img))
                #if len(final_img)>3:
                #    message = bytes(str(cv_result[3])+ "\n" + np.array2string(final_img[2],precision=0,separator=','), "utf8")
                #    print(str(message))
                #else:
                #    message = bytes(str(cv_result[3])+ "\n" + cv_result[0], "utf8")

                t2 = time.clock()
                result = 'result.xml'
                payload = open(result,'rb')

                # Send response
                self.send_response(200)
                # Send headers
                self.send_header('Content-type','text/xml')
                self.send_header('Content-length',str(os.path.getsize(result)))
                self.end_headers()

                self.wfile.write(payload.readline())
                #self.wfile.write(bytes(payload, "utf8"))
                payload.close()
            except Exception as e:
                t1 = time.clock()
                #print(e)
                t2 = time.clock()
                self.send_response(200)
                message = 'Detection Error'
                self.send_header('Content-type','text/html')
                self.end_headers()
                self.wfile.write(bytes(message, "utf8"))
            finally:
                t3 = time.clock()                
                print(detection_mode,":",(cv_time*1000), ",", (tf_detect*1000),",",(tf_io*1000),",", (t1-t0)*1000,",",(t3-t1)*1000,",",(t3-t0)*1000)
                #print("XML/Tree operation:",t2-t1)
                #print("Sending output", t3-t2)
                #print("Overall time:",t3-t0)            

        else:
            print("Content-type is ", content_type,". Should be image.")
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()

            # Send message back to client
            message = "Object not detected."
            # Write content as utf-8 data
            self.wfile.write(bytes(message, "utf8"))

        self.close_connection
        return
        #client.close()


# In[14]:


# In[4]:

def run():
    print('starting server...')
 
  # Server settings
  # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('0.0.0.0', 8081)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    print("CV_Time, TF_Detect, TF_IO, Overall_Detection, XML_Output, Overall (ms)")
    httpd.serve_forever()


# In[15]:


# In[5]:

run()

