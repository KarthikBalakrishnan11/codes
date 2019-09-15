import cv2
import imutils

cap = cv2.VideoCapture(0)
avg = None
idx=0
motionCounter = 0
minCounter=20

cv2.namedWindow('Object_Detector', cv2.WINDOW_NORMAL)

while True:
    ret,frame = cap.read()
    #text="No_Foreign_Objects"
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue
    
    cv2.accumulateWeighted(gray, avg, 0.7)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 10000:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Foreign Object Detected"

        cv2.putText(frame, text, (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        motionCounter = motionCounter+1
        if (motionCounter>=minCounter):
            idx=idx+1
            cv2.imwrite('./result/{}.jpg'.format(idx),frame)
            motionCounter=0
        # display the security feed
    cv2.imshow("Object_Detector", frame)
    key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

    # clear the stream in preparation for the next frame
    
cap.release()
cv2.destroyAllWindows()
