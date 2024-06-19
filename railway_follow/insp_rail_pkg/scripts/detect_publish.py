#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from my_custom_interfaces.msg import Object_Detection

from geometry_msgs.msg import Pose
from std_msgs.msg import String  # Import String message type for publishing
from std_msgs.msg import Bool


import argparse
import os
import sys



# Load the YOLOv8 model
model = YOLO('/home/alaaeldin/follow_railway/src/railway_follow/insp_rail_pkg/model/YoloV8_Models/result-standard/content/results/obstacles-detection/weights/best.pt')
unique_id = set()

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

class camera_1:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/output/image_raw", Image, self.callback)
        self.bool_pub = rospy.Publisher('/object_detected', Bool, queue_size=10)  # Bool message
        
    
    def callback(self, data):
        bridge = CvBridge()

        try:
            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        image = cv_image
        results = model.track(source=image, tracker="botsort.yaml", persist=True, conf=0.5, iou=0.5, show=True)
        img = results[0].plot()
        height, width, _ = img.shape
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, _id in zip(boxes, ids):
                # Check if the id is unique
                int_id = int(_id)
                if int_id not in unique_id:
                    unique_id.add(int_id)

                    # Publish a boolean message when an object is detected
                    bool_msg = Bool()
                    bool_msg.data = True
                    self.bool_pub.publish(bool_msg)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of Obstacles: {len(unique_id)}', (width - 500, 35), 0, 1, [225, 255, 255],
                    thickness=2, lineType=cv2.LINE_AA)
        resized_img = ResizeWithAspectRatio(img, height=720)
        cv2.imshow('Detected Frame', resized_img)
        cv2.waitKey(3)

def main():
    camera_1()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('camera_read', anonymous=False)
    main()