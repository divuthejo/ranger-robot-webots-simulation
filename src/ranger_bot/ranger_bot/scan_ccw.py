"""Republish /scan in standard counter-clockwise order for slam_toolbox.

Webots publishes the LDS-01 scan clockwise (angle_min = +pi, negative angle_increment).
The geometry is right, but slam_toolbox's laser model assumes increasing angles and
mapped such scans mirrored: straight fences came out curved and the pose estimate
drifted metres away from the (accurate) IMU-fused odometry.

Same beams, reversed order: angle_min <-> angle_max, increment and ranges flipped.

It also turns "no return" beams into free space. In open grass every LDS-01 beam reads
inf, and slam_toolbox drops inf beams, so the middle of the reserve stayed unknown.
Karto traces a beam as free space without an obstacle at its end when
range_threshold (max_laser_range) <= reading < the laser's maximum range, so inf
becomes exactly range_max and the advertised range_max is raised a little.

    /scan (clockwise) -> /scan_ccw (counter-clockwise, inf -> free beams)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def to_ccw(scan):
    """Return `scan` with increasing angles; scans that already increase are returned unchanged."""
    if scan.angle_increment >= 0:
        return scan
    out = LaserScan()
    out.header = scan.header
    out.angle_min = scan.angle_max
    out.angle_max = scan.angle_min
    out.angle_increment = -scan.angle_increment
    out.time_increment = scan.time_increment
    out.scan_time = scan.scan_time
    out.range_min = scan.range_min
    out.range_max = scan.range_max
    out.ranges = list(reversed(scan.ranges))
    out.intensities = list(reversed(scan.intensities))
    return out


def inf_to_free(scan, margin=0.5):
    """Report beams with no return (inf / beyond range_max) as clear up to range_max.

    The published range_max grows by `margin`, so those beams sit inside the laser's
    range but at/above slam_toolbox's max_laser_range (keep that = the sensor's range_max).
    """
    out = LaserScan()
    out.header = scan.header
    out.angle_min = scan.angle_min
    out.angle_max = scan.angle_max
    out.angle_increment = scan.angle_increment
    out.time_increment = scan.time_increment
    out.scan_time = scan.scan_time
    out.range_min = scan.range_min
    out.range_max = scan.range_max + margin
    out.ranges = [scan.range_max if (math.isinf(r) and r > 0) or r > scan.range_max else r
                  for r in scan.ranges]
    out.intensities = scan.intensities
    return out


class ScanCcw(Node):

    def __init__(self):
        super().__init__('scan_ccw')
        self.free_space = self.declare_parameter('inf_as_free_space', True).value
        self.pub = self.create_publisher(LaserScan, 'scan_ccw', qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)

    def on_scan(self, msg):
        out = to_ccw(msg)
        if self.free_space:
            out = inf_to_free(out)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanCcw()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
