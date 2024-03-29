import cv2
import imutils
import numpy as np
import time
import speech_recognition as sr
import os
# Load the MobileNetSSD model


def talk(audio):  # it is used to speak the label
    sentence = "espeak \""+audio+"\""
    os.system(sentence)  # executing the espeak module using linux terminal
     

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt , model)

# Define class labels for the MobileNetSSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# it captures the live video
cap = cv2.VideoCapture(0)


# Get the screen resolution
screen_width = 1920  # Adjust this to match your screen's width
screen_height = 1080  # Adjust this to match your screen's height

# Create a window to display the video feed in full screen
cv2.namedWindow("Object Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    
    
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to match the screen resolution
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network to get detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections and draw bounding boxes
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:  # Adjust confidence threshold as needed
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([screen_width, screen_height, screen_width, screen_height])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("Object Detection", frame)
    
    try:                # To check if the object is detected
        print(label)
        talk(label)
    except:
        print("Object Not Detected")
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) and 0xFF == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
