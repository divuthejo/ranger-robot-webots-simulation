"""Navigate the reserve on the saved map: simulation + Nav2 (map_server, AMCL, planner,
Regulated Pure Pursuit) + RViz.

    ros2 launch ranger_bot navigation.launch.py

The robot starts at the map origin, so AMCL is initialised there automatically. Send goals
with RViz's "Nav2 Goal" tool, or from code via the navigate_to_pose action.
The map comes from the mapping phase (mapping.launch.py + mapping_tour).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ACTIVATE_ROCK_MASK = '''
for i in $(seq 60); do
    state=$(ros2 lifecycle get /rock_mask_server 2>/dev/null | cut -d' ' -f1)
    case "$state" in
        active) echo "rock_mask_server active"; exit 0 ;;
        unconfigured) ros2 lifecycle set /rock_mask_server configure ;;
        inactive) ros2 lifecycle set /rock_mask_server activate ;;
    esac
    sleep 1
done
echo "rock_mask_server did not become active" >&2; exit 1
'''


def generate_launch_description():
    pkg = get_package_share_directory('ranger_bot')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    # Starts in AUTO (Nav2 drives). With teleop:=ps4 it starts in MANUAL instead; switch with
    # Triangle (auto) / Square (manual) / Circle (shared) - see ranger_bot/mode_manager.py.
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'ranger.launch.py')),
        launch_arguments={'teleop': LaunchConfiguration('teleop')}.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_pkg, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'params_file': LaunchConfiguration('params_file'),
            'use_sim_time': 'true',
            'autostart': 'true',
            'use_composition': 'False',
            'slam': 'False',
        }.items(),
    )

    # Rocks too low for the lidar: maps/keepout.yaml (from `ros2 run ranger_bot keepout_mask`)
    # is served on /rock_mask and read by a static "rock_layer" in both costmaps.
    keepout = [
        Node(package='nav2_map_server', executable='map_server', name='rock_mask_server',
             parameters=[{'use_sim_time': True,
                          'yaml_filename': LaunchConfiguration('keepout_mask'),
                          'topic_name': '/rock_mask',
                          'frame_id': 'map'}]),
        # Not nav2's lifecycle_manager: under startup load its change_state call can time out
        # ("failed to send response ... (timeout)") and it never retries, leaving the mask
        # server inactive -> rock_layer never gets a map -> every plan fails with "Costmap
        # timed out waiting for update". This retries until the server is active.
        ExecuteProcess(name='activate_rock_mask', cmd=['bash', '-c', ACTIVATE_ROCK_MASK]),
    ]

    # AMCL starts with a wide particle spread; the spawn pose is known exactly, so tighten it.
    initial_pose = Node(package='ranger_bot', executable='initial_pose',
                        parameters=[{'use_sim_time': True}])

    # What AUTO does with no other goal: loop the patrol route (ranger_bot/patrol.py).
    patrol = Node(package='ranger_bot', executable='patrol', name='patrol',
                  parameters=[{'use_sim_time': True}])

    # Tasks in AUTO: "go towards the deer", "go near the pond" (ranger_bot/tasks.py).
    tasks = Node(package='ranger_bot', executable='tasks', name='tasks',
                 parameters=[{'use_sim_time': True}])

    # Animals in the camera -> /ranger/detections, sightings log, "Deer spotted"
    # (ranger_bot/animal_detector.py; model trained with vision_dataset.py).
    # Low priority and 3 of the 12 CPU threads: with OpenVINO on every core the detector took
    # ~4 cores, load hit 35 and DDS dropped other nodes' discovery - the tracker's output
    # stopped reaching the intruder node (2026-11-01: intruder seen, no alarm).
    vision = Node(package='ranger_bot', executable='animal_detector', name='animal_detector',
                  prefix='nice -n 10 taskset -c 7-9',
                  parameters=[{'use_sim_time': True, 'rate_hz': 2.0}],
                  condition=IfCondition(LaunchConfiguration('vision')))

    # Camera boxes + lidar -> animal/person positions on the map (ranger_bot/animal_tracker.py),
    # and the intruder alarm on 'person' tracks (ranger_bot/intruder.py).
    tracker = Node(package='ranger_bot', executable='animal_tracker', name='animal_tracker',
                   parameters=[{'use_sim_time': True}],
                   condition=IfCondition(LaunchConfiguration('vision')))
    intruder = Node(package='ranger_bot', executable='intruder', name='intruder',
                    parameters=[{'use_sim_time': True}],
                    condition=IfCondition(LaunchConfiguration('vision')))
    # Makes sure Nav2's lifecycle nodes really end up ACTIVE (ranger_bot/lifecycle_guard.py):
    # the lifecycle managers give up on one service timeout, leaving AMCL/bt_navigator inactive.
    lifecycle_guard = Node(package='ranger_bot', executable='lifecycle_guard',
                           name='lifecycle_guard', parameters=[{'use_sim_time': True}])

    # Simulated battery: low -> home to the dock -> charge -> patrol again (ranger_bot/battery.py).
    battery = Node(package='ranger_bot', executable='battery', name='battery',
                   parameters=[{'use_sim_time': True}],
                   condition=IfCondition(LaunchConfiguration('battery')))

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(nav2_pkg, 'rviz', 'nav2_default_view.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('teleop', default_value='none', description="'ps4' or 'none'"),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('vision', default_value='true',
                              description='run the animal detector'),
        DeclareLaunchArgument('battery', default_value='true'),
        DeclareLaunchArgument('vision_delay', default_value='45.0',
                              description='s before the detector/tracker/intruder start'),
        DeclareLaunchArgument('map', default_value=os.path.join(pkg, 'maps', 'reserve.yaml')),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(pkg, 'config', 'nav2.yaml')),
        DeclareLaunchArgument('keepout_mask',
                              default_value=os.path.join(pkg, 'maps', 'keepout.yaml')),
        simulation,
        *keepout,
        nav2,
        initial_pose,
        patrol,
        tasks,
        # Vision starts 45 s late: loading two OpenVINO models next to Webots + Nav2 overloaded
        # the laptop (load 30+), and nodes that start during that overload sometimes never
        # connect (2026-11-01: wheel spawner never found the controller manager, the detector
        # never got camera frames). Driver, controllers and Nav2 now come up on a calm machine.
        TimerAction(period=LaunchConfiguration('vision_delay'),
                    actions=[vision, tracker, intruder]),
        TimerAction(period=30.0, actions=[lifecycle_guard]),
        battery,
        rviz,
    ])
