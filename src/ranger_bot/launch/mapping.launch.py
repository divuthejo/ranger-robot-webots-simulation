"""Build a map of the reserve: simulation + joystick + slam_toolbox + RViz.

    ros2 launch ranger_bot mapping.launch.py

Drive the whole reserve with the PS4 controller (hold L1), then save the map:

    ros2 run nav2_map_server map_saver_cli -f ~/ranger_ws/src/ranger_bot/maps/reserve \\
        --ros-args -p use_sim_time:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ranger_bot')
    slam_pkg = get_package_share_directory('slam_toolbox')

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'ranger.launch.py')),
        launch_arguments={'teleop': LaunchConfiguration('teleop')}.items(),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_pkg, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'slam_params_file': os.path.join(pkg, 'config', 'slam_toolbox.yaml'),
            'use_sim_time': 'true',
        }.items(),
    )

    # slam_toolbox needs counter-clockwise scans (see ranger_bot/scan_ccw.py).
    scan_ccw = Node(package='ranger_bot', executable='scan_ccw',
                    parameters=[{'use_sim_time': True}])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg, 'rviz', 'mapping.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('teleop', default_value='ps4', description="'ps4' or 'none'"),
        DeclareLaunchArgument('rviz', default_value='true'),
        simulation,
        scan_ccw,
        slam,
        rviz,
    ])
