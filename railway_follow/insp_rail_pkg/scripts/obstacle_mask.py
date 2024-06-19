#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import numpy as np

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

class CameraNode:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/output/image_raw", Image, self.callback)
        self.bool_pub = rospy.Publisher('/object_detected', Bool, queue_size=10)
        self.mask_pub = rospy.Publisher('/obstacle_mask', Image, queue_size=10)
        self.center_pub = rospy.Publisher('/obstacle_center', Pose, queue_size=10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        results = model.track(source=cv_image, tracker="botsort.yaml", persist=True, conf=0.5, iou=0.5, show=False)

        if results:
            result = results[0]
            if result.boxes.id is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for mask, box, _id, conf in zip(masks, boxes, ids, confidences):
                    int_id = int(_id)
                    if int_id not in unique_id:
                        unique_id.add(int_id)
                        bool_msg = Bool()
                        bool_msg.data = True
                        self.bool_pub.publish(bool_msg)

                    # Create the mask
                    obstacle_mask = (mask > 0.5).astype(np.uint8) * 255

                    # Calculate the center of the mask
                    M = cv2.moments(obstacle_mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0

                    # Publish the mask
                    mask_msg = self.bridge.cv2_to_imgmsg(obstacle_mask, encoding="mono8")
                    self.mask_pub.publish(mask_msg)

                    # Publish the center coordinates
                    center_pose = Pose()
                    center_pose.position.x = cX
                    center_pose.position.y = cY
                    center_pose.position.z = 0
                    self.center_pub.publish(center_pose)

                    rospy.loginfo(f'Obstacle ID: {_id}, Center: ({cX}, {cY})')

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                img = result.plot()
                height, width, _ = img.shape
                cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
                cv2.putText(img, f'Number of Obstacles: {len(unique_id)}', (width - 500, 35), 0, 1, [225, 255, 255],
                            thickness=2, lineType=cv2.LINE_AA)
                resized_img = ResizeWithAspectRatio(img, height=720)
                cv2.imshow('Detected Frame', resized_img)
                cv2.waitKey(3)

def main():
    rospy.init_node('camera_read', anonymous=False)
    camera_node = CameraNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
