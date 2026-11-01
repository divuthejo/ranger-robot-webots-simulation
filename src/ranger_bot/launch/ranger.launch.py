"""Wildlife ranger: Webots nature reserve + TurtleBot3 driver + PS4 joystick teleop.

    ros2 launch ranger_bot ranger.launch.py              # PS4 controller
    ros2 launch ranger_bot ranger.launch.py teleop:=none # keyboard in another terminal
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import LaunchConfigurationEquals
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def generate_launch_description():
    pkg = get_package_share_directory('ranger_bot')
    use_sim_time = {'use_sim_time': True}

    webots = WebotsLauncher(
        world=os.path.join(pkg, 'worlds', 'nature_reserve.wbt'),
        mode=LaunchConfiguration('mode'),
        ros2_supervisor=True,
    )

    # robot_state_publisher loads the full robot URDF from a file (see the header of
    # resource/ranger_description.urdf). Having the Webots driver push it at startup
    # (set_robot_state_publisher) was racy: with Nav2 starting at the same time the push
    # silently failed, so there was no LDS-01 frame. The driver's controller_manager reads
    # /robot_description too, which is why the file also carries the <ros2_control> block.
    with open(os.path.join(pkg, 'resource', 'ranger_description.urdf')) as f:
        robot_description = f.read()
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, use_sim_time],
    )
    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
        parameters=[use_sim_time],
    )

    # Wandering animals + intruder (controllers/wildlife/wildlife.py, a Webots supervisor).
    wildlife = ExecuteProcess(
        cmd=['/usr/local/webots/webots-controller', '--robot-name=wildlife',
             os.path.join(pkg, 'controllers', 'wildlife', 'wildlife.py'),
             '--intruder-every', LaunchConfiguration('intruder_every'),
             '--intruder-first', LaunchConfiguration('intruder_first'),
             ],
        output='screen',
        condition=LaunchConfigurationEquals('wildlife', 'wander'),
    )
    wildlife_still = ExecuteProcess(
        cmd=['/usr/local/webots/webots-controller', '--robot-name=wildlife',
             os.path.join(pkg, 'controllers', 'wildlife', 'wildlife.py'), '--still',
             '--intruder-every', LaunchConfiguration('intruder_every'),
             '--intruder-first', LaunchConfiguration('intruder_first')],
        output='screen',
        condition=LaunchConfigurationEquals('wildlife', 'still'),
    )

    # Jazzy's diff_drive_controller takes TwistStamped on cmd_vel.
    turtlebot_driver = WebotsController(
        robot_name='TurtleBot3Burger',
        parameters=[
            {'robot_description': os.path.join(pkg, 'resource', 'ranger_webots.urdf'),
             'use_sim_time': True,
             'set_robot_state_publisher': False},
            os.path.join(pkg, 'config', 'ros2control.yml'),
        ],
        remappings=[('/diffdrive_controller/cmd_vel', '/cmd_vel'),
                    ('/diffdrive_controller/odom', '/odom')],
        respawn=True,
    )

    # Spawn the controllers each time the driver connects to Webots. Fresh Node
    # objects are needed every time: if the driver respawns, its new controller
    # manager needs the controllers again, and a launch action cannot run twice.
    def spawn_controllers(event):
        if b'Controller successfully connected to robot in Webots simulation.' not in event.text:
            return None
        # 120 s, not 50: under a full Nav2 + voice + TTS startup the controller manager's
        # services came up too late once (2026-10-31) - no diffdrive_controller, robot stuck.
        # 2026-11-01: with RViz + both detector models starting too (load 21 on 12 cores) the
        # services took > 2 min, and two spawners also fought over the spawner lock until both
        # gave up. One spawner for both controllers, 300 s.
        spawner_args = ['--controller-manager-timeout', '300']
        return [
            Node(package='controller_manager', executable='spawner',
                 arguments=['joint_state_broadcaster', 'diffdrive_controller'] + spawner_args),
        ]
    wait_for_robot = RegisterEventHandler(
        OnProcessIO(target_action=turtlebot_driver, on_stderr=spawn_controllers))

    # Wheel odometry + IMU -> odom -> base_link TF (diff_drive's own TF is disabled).
    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[os.path.join(pkg, 'config', 'ekf.yaml')],
    )

    # Driving modes: every velocity source publishes to its own topic and mode_manager is
    # the only node on /cmd_vel (see ranger_bot/mode_manager.py):
    #   manual /cmd_vel_joy (teleop) | shared /cmd_vel_shared | auto /cmd_vel_auto (Nav2, tours)
    # PS4: Square = manual, Circle = shared, Triangle = auto; L1/R1 + stick in auto = takeover.
    ps4_config = os.path.join(pkg, 'config', 'ps4_teleop.yaml')
    joystick = [
        # SDL_JOYSTICK_HIDAPI=0: once the user can open the controller's hidraw (udev rule for
        # announcer.py's speaker), SDL switches to its HIDAPI driver - a different layout
        # (6 axes/16 buttons, L1 = 9 = our push-to-talk) and no rumble. Keep the evdev one.
        Node(package='joy', executable='joy_node', name='joy_node',
             parameters=[ps4_config, use_sim_time],
             additional_env={'SDL_JOYSTICK_HIDAPI': '0'},
             condition=LaunchConfigurationEquals('teleop', 'ps4')),
        Node(package='teleop_twist_joy', executable='teleop_node', name='teleop_twist_joy_node',
             parameters=[ps4_config, use_sim_time],
             remappings=[('cmd_vel', 'cmd_vel_joy')],
             condition=LaunchConfigurationEquals('teleop', 'ps4')),
        Node(package='ranger_bot', executable='shared_control', name='shared_control',
             parameters=[use_sim_time],
             condition=LaunchConfigurationEquals('teleop', 'ps4')),
        # Voice commands, push-to-talk on Options (voice_control.py, offline Vosk).
        Node(package='ranger_bot', executable='voice_control', name='voice_control',
             parameters=[use_sim_time],
             condition=LaunchConfigurationEquals('teleop', 'ps4')),
        # Spoken status ("Arrived at the deer", "Stopped") on the laptop speakers (announcer.py).
        Node(package='ranger_bot', executable='announcer', name='announcer',
             parameters=[use_sim_time],
             condition=LaunchConfigurationEquals('teleop', 'ps4')),
        # Controller rumble near obstacles / on bumps while a person drives (haptics.py).
        Node(package='ranger_bot', executable='haptics', name='haptics',
             parameters=[use_sim_time],
             condition=LaunchConfigurationEquals('teleop', 'ps4')),
    ]
    mode_manager = Node(
        package='ranger_bot', executable='mode_manager', name='mode_manager',
        parameters=[use_sim_time, {'initial_mode': LaunchConfiguration('drive_mode')}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='realtime', description='Webots startup mode'),
        DeclareLaunchArgument('teleop', default_value='ps4', description="'ps4' or 'none'"),
        DeclareLaunchArgument('wildlife', default_value='wander',
                              description="'wander', 'still' (static animals) or 'off'"),
        DeclareLaunchArgument('intruder_every', default_value='300',
                              description='s between intruder visits, 0 = never'),
        DeclareLaunchArgument('intruder_first', default_value='120'),
        # Start in manual with a joystick, in auto without one (Nav2 / mapping_tour drive).
        DeclareLaunchArgument(
            'drive_mode',
            default_value=PythonExpression(
                ["'manual' if '", LaunchConfiguration('teleop'), "' == 'ps4' else 'auto'"]),
            description="'manual', 'shared' or 'auto'"),
        webots,
        webots._supervisor,
        robot_state_publisher,
        footprint_publisher,
        turtlebot_driver,
        wildlife,
        wildlife_still,
        wait_for_robot,
        ekf,
        *joystick,
        mode_manager,
        # Shut everything down when the Webots window is closed.
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ])
