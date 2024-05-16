# Recommend to run the below code in Pycharm IDE or VS code or Spyder

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

# https://github.com/abewley/sort
# Download the .py file from the above given link for object tracking.
from sort import *

# Capturing the Video
# cap = cv2.VideoCapture("../videos/cars.mp4")
cap = cv2.VideoCapture("../data/source/CarParkPos old.mp4")

# Create a named window with WINDOW_NORMAL flag
cv2.namedWindow("Car Counter", cv2.WINDOW_NORMAL)

# Setting the initial width and height of the window
window_width = 1280
window_height = 720
cv2.resizeWindow("Car Counter", window_width, window_height)

# Creating instance of the model
model = YOLO("../yolo_weights/yolov8n.pt")

# Creating the list of class names as per Microsoft's COCO Dataset
classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("../data/source/mask.png")
print(f'Shape of Mask: {mask.shape}')

# Creating instance of SORT thereby tracking each object in a frame
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# let's create a line through which, if the car crosses... it will be counted
limits = [260, 350, 750, 350]

total_counts = []

while True:
    # Capturing each frame for each loop run and assigning to img variable
    success, img = cap.read()

    if not success:
        break  # exit the loop if there are no more frames to read

    print(f"Shape of Image: {img.shape}\nSuccess of Frame Rate(bool): {success}")

    img_region = cv2.bitwise_and(img, mask)

    # predict on using the trained model
    results = model(img_region, stream=True)

    detections = np.empty((0, 5))

    for detected_obj in results:
        boxes = detected_obj.boxes
        for box in boxes:
            # apply bounding-box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            print(f"x1,y1 coordinate: {(x1, y1)}\nx2,y2 coordinate: {x2, y2}\nWidth: {w}\nHeigth:{h}")

            # find out the confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(f"Confidence of box: {conf}\n", "-" * 30)

            # find out the class name
            cls = int(box.cls[0])

            # apply label/text to only car class
            current_class = classnames[cls]
            if current_class == "car" or current_class == "truck" or current_class == "bus" \
                    or current_class == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f"{current_class} {conf}", (max(0,x1+5), max(35,y1-10)), scale=1.0,
                #                    thickness=2, colorT=(0,0,0), colorR=(255,255,255), border=2, colorB=(0,0,0),
                #                    offset=8, font=cv2.FONT_HERSHEY_COMPLEX_SMALL)
                # offset = clearance around the text

                # showing the rectangle
                # cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 0, 0), rt=2, l=15)  # l = length of cornerRectangle

                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    # to track the objects, just run the below 1 line, simple!
    results_tracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)  # thickness = 5

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=3, colorR=(0, 0, 0))
        cvzone.putTextRect(img, f"({id}) {current_class} {conf}", (max(0, x1 + 5), max(35, y1 - 10)), scale=1.0,
                           thickness=2, colorT=(0, 0, 0), colorR=(255, 255, 255), border=2, colorB=(0, 0, 0),
                           offset=8, font=cv2.FONT_HERSHEY_COMPLEX_SMALL)

        # now let's find the     center point, if it touches the line, then we'll say it was the count!
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # radius=5, color, thickness=cv2.FILLED

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if total_counts.count(id) == 0:  # count the no. of times the id is present in total_counts list
                total_counts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # thickness = 5

    cvzone.putTextRect(img, f"Count:{len(total_counts)}", (20, 63), scale=2.0, thickness=2, colorT=(0, 0, 0),
                       colorR=(255, 255, 255), border=8, colorB=(0, 0, 0), offset=15, font=cv2.FONT_HERSHEY_DUPLEX)

    cv2.imshow("Car Counter", img)
    cv2.imshow("Image Region", img_region)
    if cv2.waitKey(0) == 27:
        break

print("\n", "=" * 20, f"Total Count of cars: {len(total_counts)}", "=" * 20)

cv2.destroyAllWindows()
