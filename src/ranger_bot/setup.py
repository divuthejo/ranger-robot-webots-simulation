from glob import glob

from setuptools import find_packages, setup

package_name = 'ranger_bot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/worlds', glob('worlds/*.wbt')),
        ('share/' + package_name + '/resource', glob('resource/*.urdf')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/controllers/wildlife', glob('controllers/wildlife/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divuthejo',
    maintainer_email='djdivyajoshi774@gmail.com',
    description='Wildlife ranger TurtleBot in Webots.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'shared_control = ranger_bot.shared_control:main',
            'mapping_tour = ranger_bot.mapping_tour:main',
            'scan_ccw = ranger_bot.scan_ccw:main',
            'keepout_mask = ranger_bot.keepout_mask:main',
            'initial_pose = ranger_bot.initial_pose:main',
            'mode_manager = ranger_bot.mode_manager:main',
            'patrol = ranger_bot.patrol:main',
            'haptics = ranger_bot.haptics:main',
            'voice_control = ranger_bot.voice_control:main',
            'tasks = ranger_bot.tasks:main',
            'announcer = ranger_bot.announcer:main',
            'vision_dataset = ranger_bot.vision_dataset:main',
            'animal_detector = ranger_bot.animal_detector:main',
            'task = ranger_bot.tasks:send_main',
            'animal_tracker = ranger_bot.animal_tracker:main',
            'intruder = ranger_bot.intruder:main',
            'battery = ranger_bot.battery:main',
            'lifecycle_guard = ranger_bot.lifecycle_guard:main',
        ],
    },
)
