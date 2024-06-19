#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import sensor_msgs
from std_msgs.msg import Bool  # Import Bool message type

# Variable to store the object detection status
object_detected = False

def object_detected_callback(data):
    global object_detected
    object_detected = data.data
    rospy.loginfo(f"Object Detection Status: {object_detected}")

def gps_callback(data):
    global object_detected
    if object_detected is True:
        rospy.loginfo("Object Detected - GPS Location:")
        rospy.loginfo(f"Latitude: {data.latitude}")
        rospy.loginfo(f"Longitude: {data.longitude}")
        rospy.loginfo(f"Altitude: {data.altitude}")

def gps_subscriber():
    rospy.init_node('gps_subscriber', anonymous=True)

    # Subscribe to the object detection status
    rospy.Subscriber("/object_detected", Bool, object_detected_callback)
    rospy.Subscriber("/gps_location", NavSatFix, gps_callback)
    # rospy.Subscriber("/gps_location", sensor_msgs, gps_callback)

    rospy.loginfo("GPS Subscriber node is running.")
    rospy.spin()

if __name__ == "__main__":
    gps_subscriber()
