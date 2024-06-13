import cv2
import numpy as np
from kalman_filter import kalman_filter
# from ultralytics import YOLO
import torch
import os
import cv2

# Load fine-tuned custom model
model = torch.hub.load('/home/karan/Documents/YOLOV7/yoyo/yolov7', 'custom', '/home/karan/Documents/YOLOV7/yoyo/yolov7/yolov7.pt',
                        source='local')

# Load a model
# model = YOLO('yolov7.pt')  # pretrained YOLOv8n model

kf = kalman_filter()
# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements

# Set the transition matrix
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32)

# Set the initial state
kalman.statePre = np.array([[0],
                            [0],
                            [0],
                            [0]], np.float32)

# Set the measurement matrix
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

# Set the measurement noise covariance matrix
kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1e-2

# Set the process noise covariance matrix
kalman.processNoiseCov = np.array([[1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 1, 0, 1]], np.float32) * 1e-5

# Load the video
cap = cv2.VideoCapture('/home/karan/Documents/videoplayback_6hAPpuim.mp4')
X = []
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

     # Perform object detection on the frame
    detections = model(frame)
    results = detections.pandas().xyxy[0].to_dict(orient="records")

    # Process the detections
    # results = detections.pandas().xyxy[0].to_dict(orient="records")
    
    for result in results:
        # Process each detected object
        print(result)

    # Display the frame with bounding boxes


    
       
    for result in results:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                # Do whatever you want
                x = int((x1+x2)/2)
                y = int((y1+y2)/2)
                X.append((x,y))

                frame1 = cv2.rectangle(frame, (x1, x2), (y1, y2), (0, 255, 0), 1)
                for pt in X:
                    # cv2.circle(frame1, pt, 2, (0, 255, 0), -1)
                    predicted = kf.predict(pt[0], pt[1])
                for i in range(5):
                    predicted = kf.predict(predicted[0], predicted[1])
                    cv2.circle(frame1, predicted, 2, (0, 255, 0), 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2),(0,50 ,200 ), 2)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # Perform some image processing or object detection to get the measurement      
    # In this example, we assume the measurement is the centroid of a detected object
    x, y = 100, 100  # Sample measurement values

    # Apply the Kalman filter
    measurement = np.array([[x], [y]], np.float32)
    kalman.correct(measurement)
    prediction = kalman.predict()

    # Extract the predicted coordinates
    predicted_x, predicted_y = prediction[0][0], prediction[1][0]

    # Draw the predicted coordinates on the frame
    cv2.circle(frame, (int(predicted_x), int(predicted_y)), 1, (255, 0, 0), -1)

    # Display the frames
#     cv2.imshow('Object Detection', frame)
#     cv2.imshow('Object', frame1)
    cv2.imshow('Kalman Filter', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video and close windows
cap.release()
cv2.destroyAllWindows()
