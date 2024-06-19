#!/bin/bash

xterm -e bash -c "roslaunch insp_rail_pkg view_uav_image.launch ; exec bash" &
sleep 2
# xterm -e bash -c "rosrun insp_rail_pkg img_listener_UDP.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg UDP_server.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg detect_publish.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg obstacle_mask.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg mask_show.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg gps_publisher.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg detectron2_ros.py ; exec bash" &
xterm -e bash -c "rosrun insp_rail_pkg detectron2_mask.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg obstacles_track.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg drone_controller_simulation_stop.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg drone_controller_simulation_mask.py ; exec bash" &
xterm -e bash -c "rosrun insp_rail_pkg drone_controller_simulation_original.py ; exec bash" &
# xterm -e bash -c "rosrun insp_rail_pkg command_publisher.py ; exec bash" &
xterm -e bash -c "rosrun insp_rail_pkg cmd_vel_publisher.py ; exec bash" &

echo "script finished"
