import numpy as np
import cv2
import imutils
from skimage import exposure
import pytesseract
import paho.mqtt.client as mqttClient
import json
import threading
#from picamera import PiCamera
#from picamera.array import PiRGBArray
from time import sleep
#import RPi.GPIO as GPIO
#from PyQt5 import Qt, QtGui, QtCore
#from PyQt5 import QtCore, QtGui, QtWidgets
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

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
 
broker_address= "m16.cloudmqtt.com"  
port = 13443                         
user = "etxxblbc"                    
password = "8BWOLT-KO-oX"           
 
client = mqttClient.Client("prabhaLaptop")         
client.username_pw_set(user, password=password)    
client.on_connect= on_connect                      
client.on_message= on_message                     
 
client.connect(broker_address, port=port)
cap = cv2.VideoCapture(0)

#sleep(1)

def myFunc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = exposure.rescale_intensity(gray, out_range = (0, 255))

    edged = cv2.Canny(gray, 75, 200)
    mask = cv2.dilate(edged, None, iterations=5)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        if len(approx) == 4:
            if ((approx[0][0][0]>approx[1][0][0]) and (approx[0][0][0]>approx[2][0][0])):
                (tr, tl, bl, br) = approx                    
            
                pts1 = np.float32([tl,tr,bl,br])
                pts2 = np.float32([[0,0],[434,0],[0,240],[434,240]])
                            
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(434,240))
                
                warp = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                warp = exposure.rescale_intensity(warp, out_range = (0, 255))

                roi = warp[138:138+53, 10:10+270] 
                
                ret,roi = cv2.threshold(roi,180,255,cv2.THRESH_BINARY)
                kernel = np.ones((1,1),np.uint8)
                roi = cv2.erode(roi,kernel,iterations = 1)
                cv2.imshow("Label",roi)
                text = pytesseract.image_to_string(roi)

                if (text!=''):
                    words = text.split('-')
                    with open('LoadId.json') as f:
                        data = json.load(f)
                        for key in data:
                            if(words[0]==key):
                                x={"loadId":words[0],"shelfId":words[1]}
                                client.publish(data[key],str(x))
                                print(data[key],str(x))
                                success="done"
                                return success
            else:
                (tr, tl, bl, br) = approx
                tr, tl, bl, br = br,tr,tl,bl
                
                pts1 = np.float32([tl,tr,bl,br])
                pts2 = np.float32([[0,0],[434,0],[0,240],[434,240]])
                            
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(image,M,(434,240))
                
                warp = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                warp = exposure.rescale_intensity(warp, out_range = (0, 255))

                roi = warp[138:138+53, 10:10+270] 
                
                ret,roi = cv2.threshold(roi,180,255,cv2.THRESH_BINARY)
                kernel = np.ones((1,1),np.uint8)
                roi = cv2.erode(roi,kernel,iterations = 1)
                cv2.imshow("Label",roi)
                text = pytesseract.image_to_string(roi)
                if (text!=''):
                    words = text.split('-')
                    with open('LoadId.json') as f:
                        data = json.load(f)
                        for key in data:
                            if(words[0]==key):
                                x={"loadId":words[0],"shelfId":words[1]}
                                client.publish(data[key],str(x))
                                print(data[key],str(x))
                                success="done"
                                return success


def main_function():
    print("Thread Area")
    t1 = threading.Thread(target=read_image) 
    t1.start() 
    t1.join() 
    print("Done!")   
 
def read_image():
    skip=1
    while (True):
        print("Thread1 started")
        ret, image = cap.read()      
        cv2.waitKey(1)
        image = imutils.resize(image, height = 500)
        cv2.imshow("Image",image)
        skip+=1
        if (skip%15==0):
            success=myFunc(image)
            if (success=="done"):
                print ("Ok")
                break
            
while (True):
    sleep(5)
    if True:
        print("Button Pressed")
        main_function()
    else:
        print("Button not pressed")
    

        
      

#if __name__ == "__main__": 
    #main_function()
    

cap.release()
cv2.destroyAllWindows()