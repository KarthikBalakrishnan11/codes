#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:03:54 2018

@author: ubuntu
"""

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time
#import json
#import requests

sys.path.append('./object_detection')

from utils import label_map_util

#cap = cv2.VideoCapture("http://admin:admin@192.168.1.120:80/ipcam/avc.cgi?audiostream=0")
#cap = cv2.VideoCapture("rtsp://admin:Geetha2502@192.168.1.64:554/Streaming/Channels/101")
#cap = cv2.VideoCapture("rtsp://admin:12345@192.0.0.64:554/Streaming/Channels/101")
cap = cv2.VideoCapture("./sample_videos/output.mp4")
#cap = cv2.VideoCapture("/root/Anton/Upper/sample4.mp4")
#cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.120:554/ufirststream")
#cap = cv2.VideoCapture("/root/Anton/tensorflow-repo/models/research/object_detection/turn_video/output4.mp4")
status = False
count =0
track_time = 3


model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
labels_path = os.path.join('data', 'mscoco_label_map.pbtxt')


ckpt_path = model_name + '/frozen_inference_graph.pb'
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(ckpt_path, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    print ("Graph loaded")
    tracking = False
    with tf.Session(graph=detection_graph) as sess:
        print ("Session Loaded")
        
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
       
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Loading images using openCV imread method
        #frame = cv2.imread("/root/Anton/tensorflow-repo/models/research/object_detection/test_images/image1.jpg")
        
        #Determining the loaded image height and width
        cur_frames = 0
        track_count = 0
        check =0
        track_frames = 0
        cv2.namedWindow("Camera frame",cv2.WINDOW_NORMAL)
        while True:
            
            ret,frame = cap.read()
            
            frame  = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
            
            bounding_box = (0,0,0,0)
            if ret == True and tracking == False:
                cur_frames+=1
                if not cur_frames % 3 == 0:
                    continue
                
                img_height, img_width, channels = frame.shape
                
                image_np_expanded = np.expand_dims(frame, axis=0)
                
            
                
                
                det_start_time = time.time()
                (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                det_stop_time = time.time()
                print ("Detection rate: ",det_stop_time-det_start_time)
                
                if(len(classes) > 0):
                    
                    object_index = int(classes[0][0])
                    
                    y1 = int(boxes[0][0][0]*img_height)
                    x1 = int(boxes[0][0][1]*img_width)
                    y2 = int(boxes[0][0][2]*img_height)
                    x2 = int(boxes[0][0][3]*img_width)
                    p1 = (x1,y1)
                    p2 = (x2,y2)
                    
                    if(object_index == 3 and scores[0][0]*100 > 70):
                        cv2.rectangle(frame,p1,p2,(0,255,0),3)
                        
                        print ("Detection Done")
                        
                        bounding = (x1,y1,x2-x1,y2-y1)
                        
                        rect = (x2-x1)*(y2-y1)
                        #print("Rect Area: ",rect)
                        percent = (float(rect)/(float(img_height)*float(img_width)))*100
                        #print ("Image height: ", img_height)
                        #print ("Image width: ", img_width)
                        #print "Pixel Percentage: ", percent
                        if(percent > 4):
                            
                            tracker = cv2.TrackerKCF_create()
                            #track_init_image  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                            tracker.init(frame,bounding)
                            tracking = True
                            cv2.namedWindow("TrackingWindow",cv2.WINDOW_NORMAL);
                            print(bounding)
                            print("Tracking Started")
                            continue
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.imshow("Camera frame", frame)    
            elif ret == True and tracking == True:
                track_frames+=1
                if not track_frames % 2 == 0:
                    continue
                track_start = time.time()
                rett,bounding_box = tracker.update(frame)
                print("Tracking Time:", time.time()-track_start)
                if(rett):
                    #print("Drawing bounding box")
                    pt1 = (int(bounding_box[0]),int(bounding_box[1]))
                    pt2 = (int(bounding_box[0]+bounding_box[2]),int(bounding_box[1]+bounding_box[3]))
                    cv2.rectangle(frame,pt1,pt2,(255,0,255),3)
                    rect_area = (pt1[0]-pt2[0])*(pt1[1]-pt2[1])
                    print("Rect area:", rect_area)
                    track_time = 10
                    cv2.imshow("TrackingWindow", frame)
                    cv2.waitKey(1)
                    
                else:
                    print("Tracking fading")
                    track_count = track_count + 1
                    if(track_count > track_time): 
                        track_count = 0
                        track_time = 1
                        track_frames = 0
                        tracking = False
                        #print("Car Tracking completed")
                        bounding_box = (0,0,0,0)
                        check= check+1
                        print("Car Count Exit: ",check)
                        print("\n")
                        
                        
            else:
                print("Image not loaded for Tracking and Detection")
                
            
        cv2.destroyAllWindows()
