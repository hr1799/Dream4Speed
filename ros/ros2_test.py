#!/usr/bin/env python3.10

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ChatterPublisher(Node):
    def __init__(self):
        super().__init__('chatter_publisher')  # Initialize the node
        # Create a publisher for the 'chatter' topic with String message type
        self.publisher = self.create_publisher(LaserScan, 'scan', 10)
        # Set a timer to call the publish function at a 1 Hz rate
        self.timer = self.create_timer(1.0, self.publish_message)

    def publish_message(self):
        # Create a message and set the data
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.angle_min = -3.14
        msg.angle_max = 3.14
        msg.angle_increment = 0.01
        msg.time_increment = 0.01
        msg.scan_time = 0.01
        msg.range_min = 0.0
        msg.range_max = 10.0
        msg.ranges = [1.0] * 629
        msg.intensities = [1.0] * 629
        # Publish the message
        self.publisher.publish(msg)
        self.get_logger().info('Publishing Scan message')

def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS 2 Python client library
    node = ChatterPublisher()  # Create the node
    rclpy.spin(node)  # Keep the node running
    node.destroy_node()  # Destroy the node when done
    rclpy.shutdown()  # Shut down the ROS 2 context

if __name__ == '__main__':
    main()
