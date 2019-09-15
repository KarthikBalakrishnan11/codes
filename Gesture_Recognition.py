from keras.models import load_model
from collections import deque
import numpy as np
import cv2
import time


cv2.namedWindow("output",cv2.WINDOW_NORMAL)

#class_names = ["c", "thumb", "ok", "fist", "l", "fist_moved", "palm_moved", "index", "down", "palm"]
class_names = ["palm", "fist", "C"]
# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")

model = load_model("/home/a1036006/Karthik/Hand_Gesture_Recognition/model.h5")

Q = deque(maxlen=5)

# initialize the video stream, pointer to output video file, and frame dimensions
writer = None
(W, H) = (None, None)
#cap=cv2.VideoCapture("/home/a1036006/Karthik/keras-video-classification_test2/example_clips/clip1.mp4")
cap=cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
out = cv2.VideoWriter('output/Output1.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame_width,frame_height))
# loop over frames from the video file stream

while(True):
    # read the next frame from the file
    ret,frame=cap.read()
    
    if not ret:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret,frame = cv2.threshold(frame,90,255,cv2.THRESH_BINARY)
    frame=frame/255
    frame = cv2.resize(frame, (320, 120)).astype("float32")
    

    start_time=time.time()
    frame = frame.reshape((1, 120, 320, 1))
    
    
    #preds = model.predict(np.expand_dims(frame, axis=0))[0]
    preds = model.predict(frame)
    print("Time:",time.time()-start_time)
    print("Prediction",preds)
    j=np.argmax(preds)
    Q.append(j)
    results = np.array(Q).mean(axis=0)
    
    
    result=class_names[int(results)]
    #result=class_names[j]
    
    print("Result",result)
    #jam_out=preds[0]*100
    #no_jam_out=preds[1]*100
    
    #results = np.array(Q).mean(axis=0)
    #print("Results",results)
    cv2.putText(output, result, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,3, (255, 0, 0), 15)
    
    '''if(results<58):
        cv2.putText(output, "Status: No Jam", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 12)
    else:
        cv2.putText(output, "Status: Jam", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 12)
    '''
    #out.write(output)
    cv2.imshow("output",output)
    key = cv2.waitKey(1) & 0xFF
    #cv2.waitKey(0)
    
    if key==ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()