# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 128

# import the necessary packages
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import cv2
import json
import paho.mqtt.client as mqttClient
import time


def on_connect(client, userdata, flags, rc):
 
    if rc == 0:
 
        print("Connected to broker")
 
        global Connected                
        Connected = True                
 
    else:
 
        print("Connection failed")
 
def on_message(client, userdata, message):
    print ("Message received: "  + message.payload)

Connected = False   
 
broker_address= "m14.cloudmqtt.com"  
port = 17170                         
user = "pfifosba"                    
password = "KZYPhUsAS0Re"           
 
client = mqttClient.Client("coollaptop")         
client.username_pw_set(user, password=password)    
client.on_connect= on_connect                      
client.on_message= on_message                     

client.connect(broker_address, port=port)


cv2.namedWindow("output",cv2.WINDOW_NORMAL)
# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")

model = load_model("/home/a1036006/Karthik/keras-video-classification/output/10Epochs/activity.model")

# initialize the image mean for mean subtraction along with the predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=50)


# initialize the video stream, pointer to output video file, and frame dimensions
writer = None
(W, H) = (None, None)
cap=cv2.VideoCapture("/home/a1036006/Karthik/keras-video-classification/example_clips/clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
#out = cv2.VideoWriter('output/Output3.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame_width,frame_height))
# loop over frames from the video file stream
'''start_mins='19.30'
start_min_sec=start_mins.split('.')
#print(start_min_sec[0],start_min_sec[1])
starts_in_msc=(int(start_min_sec[0])*60+int(start_min_sec[1]))*1000
#print(starts_in_msc)
cap.set(cv2.CAP_PROP_POS_MSEC,starts_in_msc)
'''



while(True):
    # read the next frame from the file
    ret,frame=cap.read()
    
    if not ret:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frame.copy()
    refPt=[(644, 66), (982, 906)]
    frame = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean

    # make predictions on the frame and then update the predictions
    # queue
    start_time=time.time()
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    print("Time:",time.time()-start_time)
    print("Prediction",preds)
    j=np.argmax(preds)
    outs=preds[j]*100
    
    Q.append(outs)
    results = np.array(Q).mean(axis=0)
    print("Results",results)
    #results_data=285-(2.85*results)
    
    min_res=70
    max_res=100
    temp_res=100/(max_res-min_res)
    results_data=temp_res*(100-results)
    if results_data>100:
        results_data=100
    percent=round((100-results)*3.5)
    percentage_text ="Jam possibility - "+str(percent)
    #cv2.putText(output, percentage_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,3, (255, 255, 255), 15)
    data = {"belt_name": "ps7-collector","congestion": {"status": "true", "value":round(results_data)}, "human_detected": {"status": "false"}}
    data_str = json.dumps(data)
    print("JSON",json.dumps(data))
    print(data_str)
    
    client.publish("BeltCongestion",str(data_str))    
    if(results>90):
        cv2.putText(output, "Status: No Jam", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 12)
    else:
        cv2.putText(output, "Status: Jam", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 12)
    
    #out.write(output)
    cv2.imshow("output",output)
    key = cv2.waitKey(1) & 0xFF
    
    if key==ord('q'):
        break


cap.release()
#out.release()
cv2.destroyAllWindows()