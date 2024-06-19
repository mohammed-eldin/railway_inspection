#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
import os

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
        self.bridge = CvBridge()
        self.track_results = []
        self.frame_id = 0
        self.result_file = '/home/alaaeldin/follow_railway/src/railway_follow/insp_rail_pkg/scripts/track_results'  # Make sure the path is correct and writable

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        image = cv_image
        results = model.track(source=image, tracker="botsort.yaml", persist=True, conf=0.5, iou=0.5, show=False)
        
        if results:
            result = results[0]
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for box, _id, conf in zip(boxes, ids, confidences):
                    int_id = int(_id)
                    if int_id not in unique_id:
                        unique_id.add(int_id)
                        bool_msg = Bool()
                        bool_msg.data = True
                        self.bool_pub.publish(bool_msg)

                    bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    self.track_results.append([self.frame_id, int_id, *bbox, -1, -1, conf])
                    rospy.loginfo(f'Added tracking result: {self.track_results[-1]}')

        self.frame_id += 1

        # Visualization and display
        img = result.plot()
        height, width, _ = img.shape
        cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of Obstacles: {len(unique_id)}', (width - 500, 35), 0, 1, [225, 255, 255],
                    thickness=2, lineType=cv2.LINE_AA)
        resized_img = ResizeWithAspectRatio(img, height=720)
        cv2.imshow('Detected Frame', resized_img)
        cv2.waitKey(3)

    def save_results(self):
        if self.track_results:
            try:
                with open(self.result_file, 'w') as f:
                    for track in self.track_results:
                        f.write(','.join(map(str, track)) + '\n')
                rospy.loginfo(f'Tracking results saved to {self.result_file}')
            except Exception as e:
                rospy.logerr(f'Failed to save tracking results: {e}')
        else:
            rospy.logwarn('No tracking results to save.')

def main():
    rospy.init_node('camera_read', anonymous=False)
    camera_node = CameraNode()
    
    rospy.on_shutdown(camera_node.save_results)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
