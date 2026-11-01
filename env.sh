# Source this in every terminal for the ranger project:  source ~/ranger_ws/env.sh
# Leaves any other venv (e.g. mlx_dashboard_env) and activates the ranger venv,
# which sees the ROS system packages plus ultralytics/openvino/anthropic.
type deactivate >/dev/null 2>&1 && deactivate
# Terminals opened from the VS Code snap inherit snap library/GTK paths, which make
# rviz2 crash (libpthread from /snap/core20). Drop them.
unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR LOCPATH
source /opt/ros/jazzy/setup.bash
source ~/ranger_ws/venv/bin/activate
# colcon writes '#!/usr/bin/python3' into node scripts, so also expose the venv's
# packages to the system interpreter.
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
[ -f ~/ranger_ws/install/setup.bash ] && source ~/ranger_ws/install/setup.bash
# DDS discovery through a Fast DDS discovery server on 127.0.0.1 (started here if not running).
# - Machine-local, so a Wi-Fi drop/reconnect can't cut new terminals off from running nodes
#   (happened 2026-10-31 and lost an unsaved map).
# - Replaces ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST: that mode probes a limited range of
#   participant ports, and with ~50 ROS processes (sim + Nav2 + ours) new nodes often never
#   found each other (2026-11-01: two fresh nodes exchanged 0 messages; the wheel spawner never
#   found the controller manager, the detector never got camera frames -> no intruder alarm).
# ROS_SUPER_CLIENT: every node sees the whole graph (ros2 node/topic list, rclpy graph calls).
unset ROS_AUTOMATIC_DISCOVERY_RANGE
export ROS_DISCOVERY_SERVER=127.0.0.1:11811
export ROS_SUPER_CLIENT=TRUE
if ! pgrep -f "fastdds discovery -i 0 -l 127.0.0.1 -p 11811" >/dev/null; then
    setsid nohup fastdds discovery -i 0 -l 127.0.0.1 -p 11811 >/tmp/ranger_discovery_server.log 2>&1 &
    sleep 1
fi

# Keep SDL on evdev for the PS4 pad: with hidraw access (70-ds4-hidraw.rules) its HIDAPI driver
# renumbers buttons (L1 -> 9) and drops rumble.
export SDL_JOYSTICK_HIDAPI=0
