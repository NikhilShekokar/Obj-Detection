import cv2

# Threshold to detect object
thres = 0.5

#img = cv2.imread("lena.png")
# Video Capture
cap = cv2.VideoCapture(0)
# Input Propid,Value
cap.set(3,640)
cap.set(4,480)

# Import the names
classNames = []

# Store the class files
classFile = 'coco.names'

# Read the class file in txt mode to list and remove spaces
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import Configuration file

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Create the default model to ip config path

net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Config default parameters
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
# Loop till the frame reads
    success,img = cap.read()
# Send img to model to op predictions
# ClasId,Boundingbox,Config
    classIds, confs, bbox = net.detect(img, confThreshold=0.5) # 50%
    print(classIds,bbox)

# Check detection
    if len(classIds) != 0:
# Loop through all the Id's found
#zip for info in a loop (flatten)
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
#src, bbox, color, thickness
            cv2.rectangle(img, box, color=(0,255,0),thickness=2)
# Print the name
# src,classname[0],box(x,y),font,scale,fcolor,thickness
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
# Display 'conf' values within bbox
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        cv2.imshow("Pic",img)

        cv2.waitKey(1)






