import cv2
import time
import numpy as np

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
# social_threshold = 150
social_threshold = 300
num_img = 0
this_time = 0
curr_time = 0

# Threshold of blue in HSV space
lower_blue = np.array([20, 35, 100])
upper_blue = np.array([180, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([70, 70, 70])
# preparing the mask to overlay

class_names = []
with open("_darknet.labels", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("http://192.168.43.42/stream.jpg")
# vc = cv2.VideoCapture(0)

net = cv2.dnn.readNet("custom-yolov4-detector_best.weights", "custom-yolov4-detector (1).cfg")
# net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")

# net = cv2.dnn.readNet("faster-rcnn/frozen_inference_graph.pb",\
#      "faster-rcnn/ssd_mobilenet_v2_coco.config")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

print("Frame rate:", cv2.CAP_PROP_FPS)

   

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()

    next_time = time.time()
    if next_time - curr_time < 1/5:
        # print(next_time - curr_time)
        continue

    curr_time = next_time

    if not grabbed:
        exit()

    # frame = cv2.flip(frame, 0)
    # frame = cv2.resize(frame, (640, 480))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    # print(frame.shape)

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # classes, scores, boxes = ([], [], []) 
    # print(classes, scores, boxes)
    # print(classes)
    end = time.time()

    start_drawing = time.time()
    list_det = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        x, y, w, h = box
        centroid = (int(x+w/2), int(y + h/2))
        list_det.append(centroid)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()

    flag = "all good"
    for i in range(len(list_det)):
        for j in range(i+1, len(list_det)):
            # print(list_det[i], list_det[j])
            distance = (list_det[i][0] - list_det[j][0])**2 + (list_det[i][1] - list_det[j][1])**2
            print(distance)
            if distance < social_threshold**2:
                flag = "not met"
                cv2.line(frame, list_det[i], list_det[j], (0, 0, 255), thickness = 2)
            else:
                cv2.line(frame, list_det[i], list_det[j], (0, 255, 0), thickness = 2)

    number_of_det = "Number of people: %d " %len(classes)
    social_distancing = "Social Distancing: " + flag
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)

    frame = cv2.resize(frame, (640, 480))
    
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, number_of_det, (0, 25+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, social_distancing, (0, 25+30*2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("other", frame)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # blue_frame = cv2.bitwise_and(frame, frame, mask = mask)
    # cv2.imshow("detections", blue_frame)
    # black_frame = cv2.bitwise_and(frame, frame, mask = black_mask)
    # cv2.imshow("detections black", black_frame)


    
    # if len(classes) > 0:
    if next_time - this_time > 0.5:
        print("yes")
        this_time = next_time
        curr_time_str = "result_competiton/" + str(this_time) + ".jpg"
        cv2.imwrite(curr_time_str, frame)