#!/usr/bin/env python3

# IMPORTS
import math
import rospy
from geometry_msgs.msg import Pose 
from my_custom_interfaces.msg import Drone_cmd
from std_msgs.msg import Float32
from std_msgs.msg import Bool

# GLOBAL VARIABLES
x = float(0)
y = float(0)
angle = float(0)
ground_distance = float(0)
rail_detected = float(0)
object_detected = False

old_x = float(0)
old_y = float(0)
old_angle = float(0)
old_ground_distance = float(0)

P_gain_yaw = 0.2
D_gain_yaw = 0
I_gain_yaw = 0

P_gain_throttle = 0.5
D_gain_throttle = 0
I_gain_throttle = 0

P_gain_pitch = 0.002
D_gain_pitch = 0
I_gain_pitch = 0

yaw_integral = 0
throttle_integral = 0
pitch_integral = 0

default_altitude = 3  # Default altitude in meters
low_altitude = 2  # Lower altitude when an obstacle is detected

# FUNCTIONs
def update_olds():
    global old_x, old_y, old_angle, old_ground_distance
    old_x = x 
    old_y = y
    old_angle = angle
    old_ground_distance = ground_distance

def update_integrals(yaw_e, throttle_e, pitch_e):
    global yaw_integral, throttle_integral, pitch_integral
    yaw_integral = yaw_integral + yaw_e
    throttle_integral = throttle_integral + throttle_e
    pitch_integral = pitch_integral + pitch_e

def object_detection_callback(msg):
    global object_detected
    object_detected = msg.data
    rospy.loginfo(f"Object detected: {object_detected}")

def callback_loc(pose):
    global x, y, angle, rail_detected
    x = pose.position.x
    y = pose.position.y
    angle = pose.orientation.z
    rail_detected = pose.orientation.w

def callback_ground(distance):
    global ground_distance
    ground_distance = distance.data

def main():
    rospy.init_node('drone_controller', anonymous=False)
    rospy.Subscriber("localization", Pose, callback_loc, queue_size=1)
    rospy.Subscriber("ground_distance", Float32, callback_ground, queue_size=1)
    rospy.Subscriber("/object_detected", Bool, object_detection_callback)
    command_pub = rospy.Publisher("command", Drone_cmd, queue_size=1)

    cmd = Drone_cmd()
    rate = rospy.Rate(20) 

    while not rospy.is_shutdown():
        if object_detected:
            # Stop the drone
            cmd.roll = 0
            cmd.pitch = 0
            cmd.yaw = 0
            cmd.throttle = P_gain_throttle * (low_altitude - ground_distance) + D_gain_throttle * (ground_distance - old_ground_distance) + I_gain_throttle * throttle_integral
            if abs(cmd.throttle) > 4:
                cmd.throttle = 4 * (abs(cmd.throttle) / cmd.throttle)
            command_pub.publish(cmd)
            rospy.loginfo("Obstacle Detected, Stopping")
        else:
            rad_angle = math.radians(angle)

            # Calculate drone commands based on sensor inputs
            if angle == 0:
                x_line = x
                y_line = 0
            else:
                m = math.tan(-math.pi/2-rad_angle)
                x_line = m * (m*x - y) / (1 + m*m)
                y_line = -(m*x - y) / (1 + m*m)

            if x_line > 0:
                x_perp = math.sqrt(x_line*x_line + y_line*y_line)
            else:
                x_perp = -math.sqrt(x_line*x_line + y_line*y_line)

            V_x = 2
            V_y = 0.001 * x_perp

            cmd.roll = V_x * math.cos(rad_angle) + V_y * math.sin(rad_angle)
            cmd.pitch = -V_x * math.sin(rad_angle) + V_y * math.cos(rad_angle)
            cmd.yaw = -10 * rad_angle
            cmd.throttle = P_gain_throttle * (default_altitude - ground_distance) + D_gain_throttle * (ground_distance - old_ground_distance) + I_gain_throttle * throttle_integral
            
            if abs(cmd.throttle) > 4:
                cmd.throttle = 4 * (abs(cmd.throttle) / cmd.throttle)

            if abs(cmd.yaw) > 30:
                cmd.yaw = 30 * (abs(cmd.yaw) / cmd.yaw)
            
            if abs(cmd.roll) > 5:
                cmd.roll = 5 * (abs(cmd.roll) / cmd.roll)

            if abs(cmd.pitch) > 5:
                cmd.pitch = 5 * (abs(cmd.pitch) / cmd.pitch)

            if rail_detected == 42:
                cmd.yaw = 0
                cmd.pitch = 0
                cmd.roll = 0

            command_pub.publish(cmd)

        update_olds()
        rate.sleep()

if __name__ == "__main__":
    main()
