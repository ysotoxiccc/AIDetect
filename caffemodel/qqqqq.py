import cv2
import numpy as np

# Load pre-trained human detection model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')


# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect humans in the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes around detected humans
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        personn = int(detections[0,0,i,1])
        if confidence > 0.5 and personn == 15 :


            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)



    # Display the output frame
    cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27:
            break


        if cv2.waitKey(1) == ord('q'):
            break


# Release video capture
cap.release()
cv2.destroyAllWindows()
